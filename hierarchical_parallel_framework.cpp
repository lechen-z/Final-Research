#include "hierarchical_parallel_framework.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <future>
#include <queue>

#ifdef __linux__
#ifdef HAVE_NUMA
#include <numa.h>
#endif
#include <sys/sysinfo.h>
#endif

namespace HierarchicalParallel {

// ============================================================================
// PerformanceModel Implementation
// ============================================================================

void PerformanceModel::initializeModels() {
    // 基于报告中的数学模型初始化各架构的性能预测函数
    models_[ArchType::SIMD_ONLY] = [this](const DatasetProfile& dataset) {
        return modelSIMDPerformance(dataset);
    };
    
    models_[ArchType::MULTICORE] = [this](const DatasetProfile& dataset) {
        return modelMulticorePerformance(dataset);
    };
    
    models_[ArchType::CLUSTER] = [this](const DatasetProfile& dataset) {
        return modelClusterPerformance(dataset);
    };
    
    models_[ArchType::GPU] = [this](const DatasetProfile& dataset) {
        return modelGPUPerformance(dataset);
    };
}

PerformanceMetrics PerformanceModel::predictPerformance(
    ArchType arch, const DatasetProfile& dataset, ComputeStage stage) const {
    
    PerformanceMetrics metrics;
    
    // 基于报告中的统一性能模型
    double base_latency = models_.at(arch)(dataset);
    
    // 根据计算阶段调整性能预测
    switch (stage) {
        case ComputeStage::TRAINING:
            metrics.latency_ms = base_latency * dataset.num_vectors / 1000.0;
            metrics.memory_usage_gb = dataset.num_vectors * dataset.dimension * 4.0 / (1024*1024*1024);
            break;
            
        case ComputeStage::ENCODING:
            metrics.latency_ms = base_latency * dataset.num_vectors / 2000.0;
            metrics.memory_usage_gb = dataset.num_vectors * 0.5 / (1024*1024*1024); // 压缩后
            break;
            
        case ComputeStage::QUERYING:
            metrics.latency_ms = base_latency * dataset.num_queries;
            metrics.throughput_qps = dataset.num_queries / (metrics.latency_ms / 1000.0);
            break;
            
        default:
            metrics.latency_ms = base_latency;
    }
    
    // 基于架构特性调整能耗估算
    switch (arch) {
        case ArchType::SIMD_ONLY:
            metrics.energy_consumption = metrics.latency_ms * hw_config_.num_cores * 0.1; // 低功耗
            break;
        case ArchType::MULTICORE:
            metrics.energy_consumption = metrics.latency_ms * hw_config_.num_cores * 0.3;
            break;
        case ArchType::CLUSTER:
            metrics.energy_consumption = metrics.latency_ms * hw_config_.num_nodes * 2.0;
            break;
        case ArchType::GPU:
            metrics.energy_consumption = metrics.latency_ms * hw_config_.num_gpus * 5.0; // 高功耗但高效
            break;
        default:
            metrics.energy_consumption = metrics.latency_ms * 1.0;
    }
    
    return metrics;
}

ArchType PerformanceModel::selectOptimalArchitecture(
    const DatasetProfile& dataset, ComputeStage stage,
    const std::vector<ArchType>& available_archs) const {
    
    ArchType best_arch = available_archs[0];
    double best_score = std::numeric_limits<double>::max();
    
    for (const auto& arch : available_archs) {
        auto metrics = predictPerformance(arch, dataset, stage);
        
        // 多目标优化：延迟 + 能耗权重
        double score = metrics.latency_ms + 0.1 * metrics.energy_consumption;
        
        // 根据数据集规模调整权重
        if (dataset.num_vectors > 1000000) {
            // 大数据集偏向集群和GPU
            if (arch == ArchType::CLUSTER || arch == ArchType::GPU) {
                score *= 0.7;
            }
        } else if (dataset.num_vectors < 10000) {
            // 小数据集偏向SIMD和多核
            if (arch == ArchType::SIMD_ONLY || arch == ArchType::MULTICORE) {
                score *= 0.8;
            }
        }
        
        if (score < best_score) {
            best_score = score;
            best_arch = arch;
        }
    }
    
    return best_arch;
}

std::unordered_map<ComputeStage, ArchType> PerformanceModel::optimizeHybridConfiguration(
    const DatasetProfile& dataset) const {
    
    std::unordered_map<ComputeStage, ArchType> config;
    std::vector<ArchType> available_archs = {ArchType::SIMD_ONLY, ArchType::MULTICORE, 
                                           ArchType::CLUSTER, ArchType::GPU};
    
    // 为每个计算阶段选择最优架构
    config[ComputeStage::TRAINING] = selectOptimalArchitecture(
        dataset, ComputeStage::TRAINING, available_archs);
    config[ComputeStage::ENCODING] = selectOptimalArchitecture(
        dataset, ComputeStage::ENCODING, available_archs);
    config[ComputeStage::QUERYING] = selectOptimalArchitecture(
        dataset, ComputeStage::QUERYING, available_archs);
    
    return config;
}

double PerformanceModel::modelSIMDPerformance(const DatasetProfile& dataset) const {
    // 基于报告中的SIMD性能模型
    double simd_width = hw_config_.has_avx512 ? 16.0 : 8.0; // AVX512 vs AVX2
    double ideal_speedup = simd_width;
    double efficiency = 0.8; // 考虑内存带宽限制
    
    double base_time = dataset.dimension * 1e-6; // 基础计算时间(ms)
    return base_time / (ideal_speedup * efficiency);
}

double PerformanceModel::modelMulticorePerformance(const DatasetProfile& dataset) const {
    // 基于报告中的多核性能模型：Amdahl定律
    double serial_fraction = 0.05; // 5%的串行部分
    double parallel_efficiency = 0.85; // 并行效率
    
    double base_time = dataset.dimension * 1e-6;
    double speedup = 1.0 / (serial_fraction + (1.0 - serial_fraction) / 
                           (hw_config_.num_cores * parallel_efficiency));
    
    return base_time / speedup;
}

double PerformanceModel::modelClusterPerformance(const DatasetProfile& dataset) const {
    // 基于报告中的分布式性能模型
    double computation_time = dataset.dimension * 1e-6 / hw_config_.num_nodes;
    double communication_overhead = std::log2(hw_config_.num_nodes) * 0.1; // AllReduce开销
    
    return computation_time + communication_overhead;
}

double PerformanceModel::modelGPUPerformance(const DatasetProfile& dataset) const {
    // 基于报告中的GPU性能模型
    if (!hw_config_.has_cuda || hw_config_.num_gpus == 0) {
        return std::numeric_limits<double>::max(); // 不可用
    }
    
    // GPU的高并行度但有启动开销
    double compute_time = dataset.dimension * 1e-8; // 高度并行
    double memory_transfer_time = dataset.dimension * 4.0 / (500.0 * 1024 * 1024); // 500GB/s带宽
    double launch_overhead = 0.01; // 10us启动开销
    
    return compute_time + memory_transfer_time + launch_overhead;
}

// ============================================================================
// HierarchicalScheduler Implementation  
// ============================================================================

HierarchicalScheduler::HierarchicalScheduler(const HardwareConfig& config)
    : hw_config_(config), current_arch_(ArchType::SIMD_ONLY) {
    perf_model_ = std::make_unique<PerformanceModel>(config);
}

HierarchicalScheduler::~HierarchicalScheduler() {
    stop();
}

void HierarchicalScheduler::start(int num_threads) {
    if (num_threads <= 0) {
        num_threads = std::min(hw_config_.num_cores, 8); // 默认线程数
    }
    
    shutdown_.store(false);
    worker_threads_.reserve(num_threads);
    
    for (int i = 0; i < num_threads; ++i) {
        worker_threads_.emplace_back(&HierarchicalScheduler::workerLoop, this);
    }
}

void HierarchicalScheduler::stop() {
    shutdown_.store(true);
    queue_cv_.notify_all();
    
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    worker_threads_.clear();
}

template<typename F, typename... Args>
auto HierarchicalScheduler::submitTask(F&& f, Args&&... args) 
    -> std::future<typename std::result_of<F(Args...)>::type> {
    
    using return_type = typename std::result_of<F(Args...)>::type;
    
    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );
    
    std::future<return_type> result = task->get_future();
    
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        if (shutdown_) {
            throw std::runtime_error("Cannot submit task to stopped scheduler");
        }
        
        task_queue_.emplace([task](){ (*task)(); });
    }
    
    queue_cv_.notify_one();
    return result;
}

void HierarchicalScheduler::switchArchitecture(ArchType new_arch) {
    if (current_arch_ != new_arch) {
        current_arch_ = new_arch;
        initializeArchitecture(new_arch);
        std::cout << "Switched to architecture: " << static_cast<int>(new_arch) << std::endl;
    }
}

void HierarchicalScheduler::adaptiveSchedule(const DatasetProfile& dataset, ComputeStage stage) {
    std::vector<ArchType> available_archs = {ArchType::SIMD_ONLY, ArchType::MULTICORE};
    
    if (hw_config_.num_nodes > 1) {
        available_archs.push_back(ArchType::CLUSTER);
    }
    if (hw_config_.has_cuda && hw_config_.num_gpus > 0) {
        available_archs.push_back(ArchType::GPU);
    }
    
    ArchType optimal_arch = perf_model_->selectOptimalArchitecture(dataset, stage, available_archs);
    switchArchitecture(optimal_arch);
}

void HierarchicalScheduler::workerLoop() {
    while (!shutdown_) {
        std::function<void()> task;
        
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this] { return shutdown_ || !task_queue_.empty(); });
            
            if (shutdown_ && task_queue_.empty()) {
                break;
            }
            
            task = std::move(task_queue_.front());
            task_queue_.pop();
        }
        
        task();
    }
}

void HierarchicalScheduler::initializeArchitecture(ArchType arch) {
    switch (arch) {
        case ArchType::SIMD_ONLY:
            // 初始化SIMD优化
            break;
        case ArchType::MULTICORE:
            // 初始化多核并行
#ifdef USE_OPENMP
            omp_set_num_threads(hw_config_.num_cores);
#endif
            break;
        case ArchType::CLUSTER:
            // 初始化MPI
#ifdef USE_MPI
            // MPI已在main中初始化
#endif
            break;
        case ArchType::GPU:
            // 初始化CUDA
#ifdef USE_CUDA
            if (hw_config_.has_cuda) {
                cudaSetDevice(0);
            }
#endif
            break;
        default:
            break;
    }
}

// ============================================================================
// HierarchicalANNSEngine Implementation
// ============================================================================

HierarchicalANNSEngine::HierarchicalANNSEngine(const HardwareConfig& config)
    : hw_config_(config), simd_initialized_(false), multicore_initialized_(false),
      cluster_initialized_(false), gpu_initialized_(false) {
    scheduler_ = std::make_unique<HierarchicalScheduler>(config);
    perf_model_ = std::make_unique<PerformanceModel>(config);
}

HierarchicalANNSEngine::~HierarchicalANNSEngine() {
    scheduler_->stop();
}

bool HierarchicalANNSEngine::initialize() {
    try {
        scheduler_->start();
        
        // 初始化各架构实现
        initializeSIMDImplementation();
        initializeMulticoreImplementation();
        
        if (hw_config_.num_nodes > 1) {
            initializeClusterImplementation();
        }
        
        if (hw_config_.has_cuda && hw_config_.num_gpus > 0) {
            initializeGPUImplementation();
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize ANNS engine: " << e.what() << std::endl;
        return false;
    }
}

void HierarchicalANNSEngine::setDatasetProfile(const DatasetProfile& profile) {
    dataset_profile_ = profile;
}

void HierarchicalANNSEngine::train(const float* training_data, size_t num_vectors, size_t dimension) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 自适应选择训练架构
    scheduler_->adaptiveSchedule(dataset_profile_, ComputeStage::TRAINING);
    
    // 提交训练任务
    auto future = scheduler_->submitTask([this, training_data, num_vectors, dimension]() {
        // 根据当前架构执行训练
        // 这里需要调用具体的PQ训练实现
        std::cout << "Training with " << num_vectors << " vectors of dimension " << dimension << std::endl;
    });
    
    future.wait();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // 记录性能指标
    PerformanceMetrics metrics;
    metrics.latency_ms = duration.count();
    recordPerformance(metrics);
}

void HierarchicalANNSEngine::search(const float* query, size_t num_queries, size_t k,
                                   std::vector<std::vector<int>>& indices,
                                   std::vector<std::vector<float>>& distances) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 自适应选择查询架构
    scheduler_->adaptiveSchedule(dataset_profile_, ComputeStage::QUERYING);
    
    // 准备结果容器
    indices.resize(num_queries);
    distances.resize(num_queries);
    
    // 提交查询任务
    auto future = scheduler_->submitTask([this, query, num_queries, k, &indices, &distances]() {
        for (size_t i = 0; i < num_queries; ++i) {
            indices[i].resize(k);
            distances[i].resize(k);
            
            // 这里需要调用具体的搜索实现
            for (size_t j = 0; j < k; ++j) {
                indices[i][j] = j; // 占位符
                distances[i][j] = j * 0.1f; // 占位符
            }
        }
    });
    
    future.wait();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    // 记录性能指标
    PerformanceMetrics metrics;
    metrics.latency_ms = duration.count() / 1000.0;
    metrics.throughput_qps = num_queries / (metrics.latency_ms / 1000.0);
    recordPerformance(metrics);
}

void HierarchicalANNSEngine::hybridSearch(const float* query, size_t num_queries, size_t k,
                                         std::vector<std::vector<int>>& indices,
                                         std::vector<std::vector<float>>& distances) {
    
    auto config = perf_model_->optimizeHybridConfiguration(dataset_profile_);
    
    std::cout << "Hybrid search using different architectures for different stages:" << std::endl;
    for (const auto& stage_config : config) {
        std::cout << "Stage " << static_cast<int>(stage_config.first) 
                  << " -> Architecture " << static_cast<int>(stage_config.second) << std::endl;
    }
    
    // 阶段1：粗选 - 使用高吞吐量架构
    scheduler_->switchArchitecture(config[ComputeStage::QUERYING]);
    
    // 阶段2：精排 - 使用高精度架构  
    scheduler_->switchArchitecture(ArchType::SIMD_ONLY);
    
    // 执行搜索
    search(query, num_queries, k, indices, distances);
}

PerformanceMetrics HierarchicalANNSEngine::getLastPerformance() const {
    std::lock_guard<std::mutex> lock(perf_mutex_);
    return perf_history_.empty() ? PerformanceMetrics() : perf_history_.back();
}

std::vector<PerformanceMetrics> HierarchicalANNSEngine::getPerformanceHistory() const {
    std::lock_guard<std::mutex> lock(perf_mutex_);
    return perf_history_;
}

void HierarchicalANNSEngine::generatePerformanceReport(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }
    
    file << "# Hierarchical Parallel ANNS Performance Report\n";
    file << "timestamp,latency_ms,throughput_qps,memory_usage_gb,energy_consumption,recall_at_k\n";
    
    std::lock_guard<std::mutex> lock(perf_mutex_);
    for (size_t i = 0; i < perf_history_.size(); ++i) {
        const auto& metrics = perf_history_[i];
        file << i << "," << metrics.latency_ms << "," << metrics.throughput_qps 
             << "," << metrics.memory_usage_gb << "," << metrics.energy_consumption 
             << "," << metrics.recall_at_k << "\n";
    }
}

void HierarchicalANNSEngine::recordPerformance(const PerformanceMetrics& metrics) {
    std::lock_guard<std::mutex> lock(perf_mutex_);
    perf_history_.push_back(metrics);
    
    // 保持历史记录在合理范围内
    if (perf_history_.size() > 10000) {
        perf_history_.erase(perf_history_.begin(), perf_history_.begin() + 1000);
    }
}

void HierarchicalANNSEngine::initializeSIMDImplementation() {
    // 初始化SIMD版本的搜索算法
    std::cout << "Initializing SIMD implementation..." << std::endl;
    simd_initialized_ = true;
}

void HierarchicalANNSEngine::initializeMulticoreImplementation() {
    // 初始化多核版本的搜索算法
    std::cout << "Initializing Multicore implementation..." << std::endl;
#ifdef USE_OPENMP
    omp_set_num_threads(hw_config_.num_cores);
#endif
    multicore_initialized_ = true;
}

void HierarchicalANNSEngine::initializeClusterImplementation() {
#ifdef USE_MPI
    // 初始化MPI版本的搜索算法
    std::cout << "Initializing Cluster implementation..." << std::endl;
    cluster_initialized_ = true;
#else
    std::cout << "MPI not available, skipping cluster initialization" << std::endl;
#endif
}

void HierarchicalANNSEngine::initializeGPUImplementation() {
#ifdef USE_CUDA
    // 初始化GPU版本的搜索算法
    std::cout << "Initializing GPU implementation..." << std::endl;
    if (hw_config_.has_cuda && hw_config_.num_gpus > 0) {
        cudaSetDevice(0);
        gpu_initialized_ = true;
    }
#else
    std::cout << "CUDA not available, skipping GPU initialization" << std::endl;
#endif
}

} // namespace HierarchicalParallel 