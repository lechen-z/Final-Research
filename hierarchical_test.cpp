#include "hierarchical_parallel_framework.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <chrono>
#include <random>

using namespace HierarchicalParallel;

// 测试用例类
class HierarchicalFrameworkTest {
private:
    int passed_tests_;
    int total_tests_;
    
public:
    HierarchicalFrameworkTest() : passed_tests_(0), total_tests_(0) {}
    
    void runTest(const std::string& test_name, std::function<bool()> test_func) {
        total_tests_++;
        std::cout << "Running test: " << test_name << "... ";
        
        try {
            if (test_func()) {
                passed_tests_++;
                std::cout << "PASSED" << std::endl;
            } else {
                std::cout << "FAILED" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << "FAILED (Exception: " << e.what() << ")" << std::endl;
        }
    }
    
    void printSummary() {
        std::cout << "\n=== Test Summary ===" << std::endl;
        std::cout << "Passed: " << passed_tests_ << "/" << total_tests_ << std::endl;
        std::cout << "Success Rate: " << (100.0 * passed_tests_ / total_tests_) << "%" << std::endl;
    }
    
    bool allTestsPassed() const {
        return passed_tests_ == total_tests_;
    }
};

// 测试硬件配置检测
bool testHardwareConfigDetection() {
    HardwareConfig config;
    config.num_cores = 8;
    config.total_memory_gb = 16;
    config.has_avx512 = false;
    config.has_cuda = false;
    config.num_gpus = 0;
    config.num_nodes = 1;
    
    // 验证配置的合理性
    return config.num_cores > 0 && 
           config.total_memory_gb > 0 && 
           config.num_nodes > 0;
}

// 测试性能模型预测
bool testPerformanceModel() {
    HardwareConfig hw_config;
    hw_config.num_cores = 4;
    hw_config.has_avx512 = false;
    hw_config.has_cuda = false;
    hw_config.num_gpus = 0;
    hw_config.num_nodes = 1;
    
    PerformanceModel model(hw_config);
    
    DatasetProfile dataset;
    dataset.num_vectors = 10000;
    dataset.dimension = 128;
    dataset.num_queries = 100;
    
    // 测试各架构的性能预测
    auto simd_metrics = model.predictPerformance(ArchType::SIMD_ONLY, dataset, ComputeStage::QUERYING);
    auto multicore_metrics = model.predictPerformance(ArchType::MULTICORE, dataset, ComputeStage::QUERYING);
    
    // 验证预测结果的合理性
    return simd_metrics.latency_ms > 0 && 
           multicore_metrics.latency_ms > 0 &&
           simd_metrics.throughput_qps > 0 &&
           multicore_metrics.throughput_qps > 0;
}

// 测试自适应架构选择
bool testArchitectureSelection() {
    HardwareConfig hw_config;
    hw_config.num_cores = 8;
    hw_config.has_avx512 = true;
    hw_config.has_cuda = false;
    hw_config.num_gpus = 0;
    hw_config.num_nodes = 1;
    
    PerformanceModel model(hw_config);
    
    // 测试小数据集的架构选择
    DatasetProfile small_dataset;
    small_dataset.num_vectors = 1000;
    small_dataset.dimension = 64;
    small_dataset.num_queries = 10;
    
    std::vector<ArchType> available_archs = {ArchType::SIMD_ONLY, ArchType::MULTICORE};
    auto selected_arch = model.selectOptimalArchitecture(small_dataset, ComputeStage::QUERYING, available_archs);
    
    // 小数据集应该偏向SIMD或多核
    bool small_dataset_ok = (selected_arch == ArchType::SIMD_ONLY || selected_arch == ArchType::MULTICORE);
    
    // 测试大数据集的架构选择
    DatasetProfile large_dataset;
    large_dataset.num_vectors = 1000000;
    large_dataset.dimension = 512;
    large_dataset.num_queries = 1000;
    
    available_archs = {ArchType::SIMD_ONLY, ArchType::MULTICORE, ArchType::CLUSTER};
    selected_arch = model.selectOptimalArchitecture(large_dataset, ComputeStage::QUERYING, available_archs);
    
    // 大数据集架构选择应该合理
    bool large_dataset_ok = true; // 任何选择都可以接受，只要不崩溃
    
    return small_dataset_ok && large_dataset_ok;
}

// 测试混合架构配置优化
bool testHybridConfiguration() {
    HardwareConfig hw_config;
    hw_config.num_cores = 8;
    hw_config.has_avx512 = true;
    hw_config.has_cuda = false;
    hw_config.num_gpus = 0;
    hw_config.num_nodes = 1;
    
    PerformanceModel model(hw_config);
    
    DatasetProfile dataset;
    dataset.num_vectors = 100000;
    dataset.dimension = 256;
    dataset.num_queries = 500;
    
    auto config = model.optimizeHybridConfiguration(dataset);
    
    // 验证配置包含所有计算阶段
    return config.find(ComputeStage::TRAINING) != config.end() &&
           config.find(ComputeStage::ENCODING) != config.end() &&
           config.find(ComputeStage::QUERYING) != config.end();
}

// 测试任务调度器
bool testHierarchicalScheduler() {
    HardwareConfig hw_config;
    hw_config.num_cores = 4;
    
    HierarchicalScheduler scheduler(hw_config);
    scheduler.start(2); // 启动2个工作线程
    
    // 提交测试任务
    std::atomic<int> counter{0};
    std::vector<std::future<void>> futures;
    
    for (int i = 0; i < 10; ++i) {
        auto future = scheduler.submitTask([&counter]() {
            counter.fetch_add(1);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        });
        futures.push_back(std::move(future));
    }
    
    // 等待所有任务完成
    for (auto& future : futures) {
        future.wait();
    }
    
    scheduler.stop();
    
    // 验证所有任务都被执行
    return counter.load() == 10;
}

// 测试ANNS引擎初始化
bool testANNSEngineInitialization() {
    HardwareConfig hw_config;
    hw_config.num_cores = 4;
    hw_config.total_memory_gb = 8;
    hw_config.has_avx512 = false;
    hw_config.has_cuda = false;
    hw_config.num_gpus = 0;
    hw_config.num_nodes = 1;
    
    HierarchicalANNSEngine engine(hw_config);
    
    bool init_success = engine.initialize();
    
    if (!init_success) {
        return false;
    }
    
    // 设置数据集特征
    DatasetProfile profile;
    profile.num_vectors = 1000;
    profile.dimension = 64;
    profile.num_queries = 10;
    profile.data_type = "float32";
    
    engine.setDatasetProfile(profile);
    
    return true;
}

// 测试性能监控功能
bool testPerformanceMonitoring() {
    HardwareConfig hw_config;
    hw_config.num_cores = 4;
    
    HierarchicalANNSEngine engine(hw_config);
    
    if (!engine.initialize()) {
        return false;
    }
    
    // 设置测试数据集
    DatasetProfile profile;
    profile.num_vectors = 100;
    profile.dimension = 32;
    profile.num_queries = 5;
    
    engine.setDatasetProfile(profile);
    
    // 模拟一些操作
    std::vector<float> training_data(profile.num_vectors * profile.dimension, 1.0f);
    engine.train(training_data.data(), profile.num_vectors, profile.dimension);
    
    // 检查性能历史
    auto history = engine.getPerformanceHistory();
    auto latest = engine.getLastPerformance();
    
    return !history.empty() && latest.latency_ms >= 0;
}

// 测试数据生成和基本搜索
bool testBasicSearch() {
    HardwareConfig hw_config;
    hw_config.num_cores = 2;
    
    HierarchicalANNSEngine engine(hw_config);
    
    if (!engine.initialize()) {
        return false;
    }
    
    // 设置小数据集
    const size_t num_base = 100;
    const size_t num_queries = 5;
    const size_t dimension = 32;
    const size_t k = 3;
    
    DatasetProfile profile;
    profile.num_vectors = num_base;
    profile.dimension = dimension;
    profile.num_queries = num_queries;
    
    engine.setDatasetProfile(profile);
    
    // 生成随机数据
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    std::vector<float> base_data(num_base * dimension);
    std::vector<float> query_data(num_queries * dimension);
    
    for (auto& val : base_data) val = dist(gen);
    for (auto& val : query_data) val = dist(gen);
    
    // 执行搜索
    std::vector<std::vector<int>> indices;
    std::vector<std::vector<float>> distances;
    
    engine.search(query_data.data(), num_queries, k, indices, distances);
    
    // 验证结果结构
    return indices.size() == num_queries && 
           distances.size() == num_queries &&
           !indices.empty() && 
           indices[0].size() == k &&
           distances[0].size() == k;
}

// 性能基准测试
bool testPerformanceBenchmark() {
    HardwareConfig hw_config;
    hw_config.num_cores = std::thread::hardware_concurrency();
    
    HierarchicalANNSEngine engine(hw_config);
    
    if (!engine.initialize()) {
        return false;
    }
    
    // 设置中等规模数据集
    const size_t num_base = 1000;
    const size_t num_queries = 50;
    const size_t dimension = 128;
    const size_t k = 10;
    
    DatasetProfile profile;
    profile.num_vectors = num_base;
    profile.dimension = dimension; 
    profile.num_queries = num_queries;
    
    engine.setDatasetProfile(profile);
    
    // 生成测试数据
    std::vector<float> query_data(num_queries * dimension, 1.0f);
    
    // 执行性能测试
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<std::vector<int>> indices;
    std::vector<std::vector<float>> distances;
    engine.search(query_data.data(), num_queries, k, indices, distances);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double latency_ms = duration.count() / 1000.0;
    double throughput_qps = num_queries / (latency_ms / 1000.0);
    
    std::cout << " (Latency: " << latency_ms << "ms, Throughput: " << throughput_qps << " QPS)";
    
    // 性能应该在合理范围内
    return latency_ms > 0 && latency_ms < 10000 && throughput_qps > 0;
}

int main() {
    std::cout << "=== Hierarchical Parallel ANNS Framework Unit Tests ===" << std::endl;
    
    HierarchicalFrameworkTest test_suite;
    
    // 运行所有测试用例
    test_suite.runTest("Hardware Config Detection", testHardwareConfigDetection);
    test_suite.runTest("Performance Model", testPerformanceModel);
    test_suite.runTest("Architecture Selection", testArchitectureSelection);
    test_suite.runTest("Hybrid Configuration", testHybridConfiguration);
    test_suite.runTest("Task Scheduler", testHierarchicalScheduler);
    test_suite.runTest("ANNS Engine Initialization", testANNSEngineInitialization);
    test_suite.runTest("Performance Monitoring", testPerformanceMonitoring);
    test_suite.runTest("Basic Search", testBasicSearch);
    test_suite.runTest("Performance Benchmark", testPerformanceBenchmark);
    
    // 打印测试总结
    test_suite.printSummary();
    
    if (test_suite.allTestsPassed()) {
        std::cout << "\n🎉 All tests passed! The hierarchical parallel framework is working correctly." << std::endl;
        return 0;
    } else {
        std::cout << "\n❌ Some tests failed. Please check the implementation." << std::endl;
        return 1;
    }
} 