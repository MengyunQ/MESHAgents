import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import fdrcorrection

def load_and_preprocess_data(file_path):
    """Load and preprocess data"""
    df = pd.read_csv(file_path)
    
    # Basic data cleaning
    df = df.replace([np.inf, -np.inf], np.nan)
    
    return df

def calculate_basic_statistics(data):
    """Calculate basic statistics"""
    stats_dict = {
        'mean': np.mean(data),
        'std': np.std(data),
        'median': np.median(data),
        'q1': np.percentile(data, 25),
        'q3': np.percentile(data, 75),
        'skewness': stats.skew(data),
        'kurtosis': stats.kurtosis(data)
    }
    return stats_dict

def perform_statistical_tests(data1, data2, test_type='t-test'):
    """Perform statistical tests"""
    if test_type == 't-test':
        stat, p_value = stats.ttest_ind(data1, data2)
    elif test_type == 'mann_whitney':
        stat, p_value = stats.mannwhitneyu(data1, data2)
    return stat, p_value

def calculate_effect_size(data1, data2):
    """Calculate effect size"""
    cohen_d = (np.mean(data1) - np.mean(data2)) / np.sqrt((np.var(data1) + np.var(data2)) / 2)
    return cohen_d

def apply_multiple_testing_correction(p_values, method='fdr'):
    """Multiple testing correction"""
    if method == 'fdr':
        rejected, corrected_p_values = fdrcorrection(p_values)
    elif method == 'bonferroni':
        corrected_p_values = np.minimum(p_values * len(p_values), 1.0)
        rejected = corrected_p_values < 0.05
    return rejected, corrected_p_values

def create_visualization(data, plot_type='boxplot', **kwargs):
    """Create visualization"""
    plt.figure(figsize=(10, 6))
    
    if plot_type == 'boxplot':
        sns.boxplot(data=data, **kwargs)
    elif plot_type == 'violin':
        sns.violinplot(data=data, **kwargs)
    elif plot_type == 'histogram':
        sns.histplot(data=data, **kwargs)
    
    plt.tight_layout()
    return plt.gcf()

def save_results(results, file_path):
    """Save results"""
    if isinstance(results, pd.DataFrame):
        results.to_csv(file_path)
    else:
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=4)

def generate_report(results, template_path=None):
    """Generate analysis report"""
    report = {
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'summary': results['summary'],
        'detailed_results': results['detailed_results'],
        'visualizations': results['visualizations']
    }
    return report