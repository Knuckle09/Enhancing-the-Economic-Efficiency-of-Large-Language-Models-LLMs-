import React from 'react';

// --- MOCK DEPENDENCIES START ---
// Replacing external icons with simple, styled placeholders
const TrendingDown = ({ className }) => <span className={`${className} text-xl mr-1`}>↓</span>;
const CheckCircle = ({ className }) => <span className={`${className} text-sm mr-1`}>✓</span>;
const Activity = ({ className }) => <span className={`${className} text-sm mr-1`}>⚠</span>;

// Replacing Recharts components with a simple visual placeholder
// In a real application, you would need the full Recharts library.
const ResponsiveContainer = ({ children }) => <div style={{ height: '100%', minHeight: '200px' }}>{children}</div>;
const MockChart = ({ performanceData }) => (
    <div className="flex flex-col space-y-4 pt-4">
        {performanceData.map((item, index) => (
            <div key={index} className="flex flex-col">
                <span className="text-gray-400 text-sm font-medium">{item.metric}</span>
                <div className="flex items-center mt-1">
                    <div className="h-4 w-full bg-gray-700 rounded-full">
                        <div
                            className="h-4 rounded-full transition-all duration-500"
                            style={{
                                width: `${Math.min(item.actual, 100)}%`, // Cap visual width at 100%
                                backgroundColor: item.metric === 'Token Reduction' ? '#10b981' : '#6366f1' // Green for reduction, Purple for similarity
                            }}
                        ></div>
                    </div>
                    <span className="ml-4 text-gray-200 font-bold text-sm">
                        {item.actual.toFixed(1)}%
                    </span>
                </div>
                <span className="text-xs text-gray-500 mt-1">
                    Target: {item.target.toFixed(1)}%
                </span>
            </div>
        ))}
    </div>
);
// --- MOCK DEPENDENCIES END ---


const StrategyPerformanceChart = () => {
    // Hardcoded mock data to ensure the chart renders and passes the initial check.
    const data = {
        token_reduction_percent: 42.5,
        similarity: 0.91, // 91%
        strategy_targets: {
            target_reduction: 30,
            min_similarity: 0.85, // 85%
        },
        strategy_used: 'Context-Aware Summarization',
    };

    // The component's original rendering logic starts here
    // Note: The initial check is now implicitly passed because 'data' is defined internally.
    // if (!data || !data.token_reduction_percent) return null;

    // Default targets
    const targetReduction = data.strategy_targets?.target_reduction || 30;
    const minSimilarity = data.strategy_targets?.min_similarity || 0.85;

    const performanceData = [
        {
            metric: 'Token Reduction',
            actual: data.token_reduction_percent,
            target: targetReduction
        },
        {
            metric: 'Similarity',
            actual: data.similarity * 100,
            target: minSimilarity * 100
        }
    ];

    return (
        <div className="bg-gray-800 rounded-xl shadow-2xl p-6 mb-8 border-t-4 border-purple-600">
            {/* Header Section */}
            <div className="flex items-center justify-between mb-6">
                <div className="flex items-center space-x-2">
                    <TrendingDown className="w-6 h-6 text-purple-400" />
                    <h3 className="text-xl font-semibold text-gray-200">Strategy Performance</h3>
                </div>
                <span className="bg-purple-500/20 text-purple-300 px-3 py-1 rounded-full text-sm font-medium">
                    {data.strategy_used || 'N/A'}
                </span>
            </div>
            
            {/* Chart Section (Using MockChart) */}
            <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                    <MockChart performanceData={performanceData} />
                </ResponsiveContainer>
            </div>
            
            {/* Metric Comparison Cards */}
            <div className="mt-4 grid grid-cols-2 gap-4">
                {/* Token Reduction Card */}
                <div className="bg-gray-700/50 rounded-lg p-3">
                    <div className="flex items-center justify-between">
                        <span className="text-gray-400 text-sm">Token Reduction</span>
                        {data.token_reduction_percent >= targetReduction * 0.9 ? (
                            <CheckCircle className="w-4 h-4 text-green-400" />
                        ) : (
                            <Activity className="w-4 h-4 text-yellow-400" />
                        )}
                    </div>
                    <p className="text-blue-400 font-bold text-lg mt-1">
                        {data.token_reduction_percent}% / {targetReduction}%
                    </p>
                </div>
                
                {/* Similarity Card */}
                <div className="bg-gray-700/50 rounded-lg p-3">
                    <div className="flex items-center justify-between">
                        <span className="text-gray-400 text-sm">Similarity</span>
                        {data.similarity >= minSimilarity ? (
                            <CheckCircle className="w-4 h-4 text-green-400" />
                        ) : (
                            <Activity className="w-4 h-4 text-yellow-400" />
                        )}
                    </div>
                    <p className="text-green-400 font-bold text-lg mt-1">
                        {(data.similarity * 100).toFixed(1)}% / {(minSimilarity * 100).toFixed(0)}%
                    </p>
                </div>
            </div>
        </div>
    );
};

export default StrategyPerformanceChart;