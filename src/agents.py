import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.multitest import multipletests
from statsmodels.api import OLS

from typing import Dict, List, Any, Optional, Tuple
from openai import OpenAI
import logging

from config import (
    OPENAI_API_KEY, DATA_PATH, RESULTS_PATH, LOG_PATH, DISEASES, GPT_MODEL
)

logger = logging.getLogger(__name__)

import asyncio
from scipy import stats



class Memory:
    """Agent memory system for storing analysis results and important findings"""

    def __init__(self):
        self.historical_analyses: List[Dict] = []
        self.phenotype_correlations: Dict = {}
        self.important_patterns: Dict = {}
        self.statistical_findings: Dict = {}

    def store_analysis(self, analysis_type: str, results: Dict[str, Any]) -> None:
        """Store analysis results
        
        Args:
            analysis_type: Identifier for analysis type
            results: Dictionary of analysis results
        """
        self.historical_analyses.append({
            'type': analysis_type,
            'results': results,
            'timestamp': pd.Timestamp.now()
        })

    def get_previous_analyses(self, analysis_type: Optional[str] = None) -> List[Dict]:
        """Get historical analysis results
        
        Args:
            analysis_type: Optional, specific analysis type identifier
            
        Returns:
            List of historical analyses of specified type or all types
        """
        if analysis_type:
            return [x for x in self.historical_analyses if x['type'] == analysis_type]
        return self.historical_analyses

    def store_correlation(self, phenotype1: str, phenotype2: str, 
                         correlation: float, p_value: float) -> None:
        """Store correlation between phenotypes
        
        Args:
            phenotype1: Name of the first phenotype
            phenotype2: Name of the second phenotype
            correlation: Correlation coefficient
            p_value: Significance p-value
        """
        key = f"{phenotype1}_{phenotype2}"
        self.phenotype_correlations[key] = {
            'correlation': correlation,
            'p_value': p_value,
            'timestamp': pd.Timestamp.now()
        }

    def store_pattern(self, pattern_type: str, pattern_data: Dict[str, Any]) -> None:
        """Store identified important patterns
        
        Args:
            pattern_type: Pattern type identifier
            pattern_data: Data related to the pattern
        """
        if pattern_type not in self.important_patterns:
            self.important_patterns[pattern_type] = []
        self.important_patterns[pattern_type].append({
            'data': pattern_data,
            'timestamp': pd.Timestamp.now()
        })

    def store_finding(self, finding_type: str, finding_data: Dict[str, Any]) -> None:
        """Store statistical findings
        
        Args:
            finding_type: Finding type identifier
            finding_data: Data related to the finding
        """
        if finding_type not in self.statistical_findings:
            self.statistical_findings[finding_type] = []
        self.statistical_findings[finding_type].append({
            'data': finding_data,
            'timestamp': pd.Timestamp.now()
        })

    def get_latest_findings(self, finding_type: Optional[str] = None) -> Dict:
        """Get latest statistical findings
        
        Args:
            finding_type: Optional, specific finding type identifier
            
        Returns:
            Dictionary of latest statistical findings
        """
        if finding_type:
            findings = self.statistical_findings.get(finding_type, [])
            return findings[-1] if findings else {}
        return {k: v[-1] if v else {} for k, v in self.statistical_findings.items()}

class StatisticalTools:
    """Statistical analysis tools containing various statistical methods"""

    @staticmethod
    def correlation_analysis(data: pd.DataFrame, 
                            method: str = 'pearson') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Correlation analysis"""
        if method == 'pearson':
            # Create correlation and p-value matrices
            columns = data.columns
            corr_matrix = pd.DataFrame(np.eye(len(columns)), 
                                    index=columns, 
                                    columns=columns)
            pvals = corr_matrix.copy()

            for i, col1 in enumerate(columns):
                for j, col2 in enumerate(columns):
                    if i != j:
                        # Align data for common non-null indices
                        common_data = data[[col1, col2]].dropna()

                        # Compute correlation only if sufficient data
                        if len(common_data) > 2:
                            try:
                                corr, p = pearsonr(common_data[col1], common_data[col2])
                                corr_matrix.iloc[i,j] = corr
                                pvals.iloc[i,j] = p
                            except Exception as e:
                                # Log or handle potential correlation computation errors
                                logger.warning(f"Correlation error for {col1} and {col2}: {e}")

            return corr_matrix, pvals

        elif method == 'spearman':
            return data.corr(method='spearman'), None

        else:
            raise ValueError("Method must be either 'pearson' or 'spearman'")

    @staticmethod
    def pca_analysis(data: pd.DataFrame) -> Dict[str, Any]:
        """Principal Component Analysis
        
        Args:
            data: Input dataframe
            
        Returns:
            Dictionary containing PCA results
        """
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)

        # Initialize PCA
        pca = PCA()
        pca_result = pca.fit_transform(scaled_data)

        # Calculate cumulative explained variance ratio
        cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

        # Calculate number of components needed
        n_components_95 = np.argmax(cumulative_variance_ratio >= 0.95) + 1

        return {
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance_ratio': cumulative_variance_ratio,
            'components': pca.components_,
            'transformed_data': pca_result,
            'n_components_95': n_components_95,
            'feature_weights': pd.DataFrame(
                pca.components_,
                columns=data.columns,
                index=[f'PC{i+1}' for i in range(len(pca.components_))]
            )
        }

    @staticmethod
    def feature_importance(X: pd.DataFrame, 
                         y: pd.Series,
                         n_estimators: int = 100) -> pd.Series:
        """Calculate feature importance using Random Forest
        
        Args:
            X: Feature matrix
            y: Target variable
            n_estimators: Number of trees in the forest
            
        Returns:
            Series of feature importance scores
        """
        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        rf.fit(X, y)

        importance_scores = pd.Series(
            rf.feature_importances_,
            index=X.columns,
            name='importance_score'
        ).sort_values(ascending=False)

        return importance_scores

    @staticmethod
    def multicollinearity_test(X: pd.DataFrame) -> pd.DataFrame:
        """Multicollinearity test
        
        Args:
            X: Feature matrix
            
        Returns:
            DataFrame of VIF values
        """
        X_with_constant = sm.add_constant(X)
        vif_data = pd.DataFrame()
        vif_data["Variable"] = X_with_constant.columns
        vif_data["VIF"] = [variance_inflation_factor(X_with_constant.values, i) 
                          for i in range(X_with_constant.shape[1])]

        return vif_data.sort_values('VIF', ascending=False)

    @staticmethod
    def outlier_detection(data: pd.Series,
                         method: str = 'zscore',
                         threshold: float = 3.0) -> pd.Series:
        """Outlier detection
        
        Args:
            data: Input data series
            method: Detection method, 'zscore' or 'iqr'
            threshold: Threshold (used for zscore method)
            
        Returns:
            Boolean series, True indicates outlier
        """
        if method == 'zscore':
            z_scores = np.abs(stats.zscore(data))
            return pd.Series(z_scores > threshold, index=data.index)

        elif method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            return ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR)))

        else:
            raise ValueError("Method must be either 'zscore' or 'iqr'")

    @staticmethod
    def descriptive_stats(data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate descriptive statistics
        
        Args:
            data: Input dataframe
            
        Returns:
            Dictionary of descriptive statistics results
        """
        stats_dict = {}

        for column in data.columns:
            col_data = data[column].dropna()

            if col_data.dtype in ['int64', 'float64']:
                stats_dict[column] = {
                    'basic_stats': col_data.describe().to_dict(),
                    'skewness': col_data.skew(),
                    'kurtosis': col_data.kurtosis(),
                    'missing_ratio': data[column].isna().mean(),
                    'distribution_test': stats.normaltest(col_data) if len(col_data) >= 8 else None
                }

        return stats_dict


class BaseOrganAgent:
    """Base class for organ agents"""

    def __init__(self, name: str, phenotypes: List[str], api_key: str):
        self.name = name
        self.phenotypes = phenotypes
        self.memory = Memory()
        self.tools = StatisticalTools()

        # GPT Configuration
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
        self.model = GPT_MODEL
        self.messages = []

    async def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """General analysis pipeline"""
        raise NotImplementedError

    def get_gpt_analysis(self, data_description: str, 
                              temperature: float = 0) -> str:
        """Get GPT analysis results"""
        try:
            self.messages.append({
                "role": "user", 
                "content": f"Analyse the following {self.name} data:\n{data_description}"
            })


            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages
            )

            reply = response.choices[0].message.content
            self.messages.append({"role": "assistant", "content": reply})
            return reply

        except Exception as e:
            logger.error(f"Error in GPT API call: {e}")
            raise

    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data"""
        missing_cols = set(self.phenotypes) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    def store_results(self, analysis_type: str, results: Dict[str, Any]) -> None:
        """Store analysis results"""
        self.memory.store_analysis(analysis_type, results)

    def get_phenotype_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Get relevant phenotype data"""
        return data[self.phenotypes].copy()

    def get_previous_analyses(self, analysis_type: str = None) -> List[Dict]:
        """Get historical analyses"""
        return self.memory.get_previous_analyses(analysis_type)

    async def explain_results(self, results: Dict[str, Any]) -> str:
        """Use GPT to explain analysis results"""
        explanation_prompt = f"""
        Please explain the following {self.name} analysis results, focusing on:
        1. Key findings and anomalies
        2. Potential clinical significance
        3. Suggestions for further analysis
        
        Analysis Results:
        {results}
        """
        return self.get_gpt_analysis(explanation_prompt)

    def _calculate_basic_stats(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate basic statistics"""
        return {
            'descriptive_stats': self.tools.descriptive_stats(data),
            'correlations': self.tools.correlation_analysis(data)[0].to_dict(),
            'outliers': {
                col: self.tools.outlier_detection(data[col]).sum()
                for col in data.columns
            }
        }

class LVAgent(BaseOrganAgent):
    """Left Ventricle Analysis Agent"""

    def __init__(self, api_key: str):
        phenotypes = [
            'LVEDV (mL)', 'LVESV (mL)', 'LVSV (mL)',
            'LVEF (%)', 'LVCO (L/min)', 'LVM (g)'
        ]

        super().__init__('LV', phenotypes, api_key)

        # LV-specific prior knowledge
        self.normal_ranges = {
            'LVEDV (mL)': (65, 240),  # Normal range for end-diastolic volume
            'LVEF (%)': (52, 72),     # Normal range for ejection fraction
            'LVM (g)': (88, 224)      # Normal range for LV mass
        }

        # Set specialized system prompt for LV analysis
        self.system_prompt = """You are an expert cardiologist specialized in Left Ventricular (LV) analysis. Your role is to provide comprehensive analysis of LV imaging data with the following key responsibilities:

1. Volume Analysis:
   - Evaluate End-Diastolic Volume (LVEDV) and End-Systolic Volume (LVESV)
   - Assess Stroke Volume (LVSV) and its clinical implications
   - Identify volume-related abnormalities and their potential causes

2. Systolic Function:
   - Analyze Ejection Fraction (LVEF) as a primary indicator of systolic function
   - Evaluate Cardiac Output (LVCO) and its physiological significance
   - Detect systolic dysfunction patterns

3. Mass Analysis:
   - Assess LV Mass (LVM) and its relationship to cardiac remodeling
   - Evaluate mass-to-volume ratios
   - Identify hypertrophy patterns

4. Clinical Integration:
   - Connect findings to potential pathological conditions
   - Suggest additional analyses when abnormalities are detected
   - Provide evidence-based insights for clinical decision-making

5. Data Quality:
   - Identify measurement artifacts or inconsistencies
   - Validate measurements against established normal ranges
   - Flag technically questionable results

Guidelines:
- Base analyses on established clinical guidelines and latest research
- Provide quantitative assessments with clear reference ranges
- Highlight both primary findings and subtle patterns
- Consider age and sex-specific variations in normal ranges
- Maintain awareness of measurement limitations and potential artifacts"""

        self.messages = [{"role": "system", "content": self.system_prompt}]

        # LV-specific analysis tools
        self.lv_tools = {
            'volume_analysis': self._analyze_volumes,
            'mass_analysis': self._analyze_mass,
            'ejection_analysis': self._analyze_ejection
        }

    async def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Execute LV-specific analysis pipeline"""
        self._validate_data(data)
        results = {}

        # 1. Volume Analysis
        volume_results = self.lv_tools['volume_analysis'](data)
        results['volumes'] = volume_results
        self.store_results('volume_analysis', volume_results)

        # 2. Mass Analysis
        mass_results = self.lv_tools['mass_analysis'](data)
        results['mass'] = mass_results
        self.store_results('mass_analysis', mass_results)

        # 3. Ejection Analysis
        ef_results = self.lv_tools['ejection_analysis'](data)
        results['ejection'] = ef_results
        self.store_results('ejection_analysis', ef_results)

        # 4. Statistical Analysis
        lv_data = self.get_phenotype_data(data)
        corr_matrix, pvals = self.tools.correlation_analysis(lv_data)
        results['correlations'] = {
            'correlation_matrix': corr_matrix.to_dict(),
            'p_values': pvals.to_dict()
        }

        # 5. PCA Analysis
        pca_results = self.tools.pca_analysis(lv_data.dropna())
        results['pca'] = pca_results

        # 6. GPT Analysis
        gpt_description = f"""
        Key LV measurements:
        - LVEDV: {data['LVEDV (mL)'].describe().to_dict()}
        - LVEF: {data['LVEF (%)'].describe().to_dict()}
        - LVM: {data['LVM (g)'].describe().to_dict()}
        
        Significant correlations: {results['correlations']}
        """
        results['gpt_analysis'] = self.get_gpt_analysis(gpt_description)

        return results

    def _analyze_volumes(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze LV volumes"""
        volume_data = data[['LVEDV (mL)', 'LVESV (mL)', 'LVSV (mL)']]

        # Calculate basic stats and volume-derived metrics
        volume_stats = self._calculate_basic_stats(volume_data)

        # Add volume-specific analyses
        volume_stats.update({
            'volume_ratios': {
                'esv_to_edv': (data['LVESV (mL)'] / data['LVEDV (mL)']).describe().to_dict(),
                'sv_to_edv': (data['LVSV (mL)'] / data['LVEDV (mL)']).describe().to_dict()
            },
            'abnormal_volumes': {
                'high_edv': (data['LVEDV (mL)'] > self.normal_ranges['LVEDV (mL)'][1]).sum(),
                'low_edv': (data['LVEDV (mL)'] < self.normal_ranges['LVEDV (mL)'][0]).sum()
            }
        })

        return volume_stats

    def _analyze_mass(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze LV mass"""
        # Get histogram values and bins separately
        hist_values, hist_bins = np.histogram(data['LVM (g)'].dropna(), bins=20)

        mass_stats = {
            'mass_basic': data['LVM (g)'].describe().to_dict(),
            'mass_distribution': {
                'values': hist_values.tolist(),
                'bins': hist_bins.tolist()
            }
        }

        # Mass indexing
        if 'Height' in data.columns and 'Weight' in data.columns:
            mass_stats.update({
                'mass_indexed': {
                    'height_indexed': (data['LVM (g)'] / (data['Height'] ** 2.7)).describe().to_dict(),
                    'bsa_indexed': (
                        data['LVM (g)'] / 
                        ((data['Weight'] ** 0.425) * (data['Height'] ** 0.725) * 0.007184)
                    ).describe().to_dict()
                }
            })

        # Add mass-volume relations
        mass_stats['mass_volume_ratio'] = (
            data['LVM (g)'] / data['LVEDV (mL)']).describe().to_dict()

        return mass_stats

    def _analyze_ejection(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze ejection fraction and related metrics"""
        hist_values, hist_bins = np.histogram(data['LVEF (%)'].dropna(), bins=20)

        ef_stats = {
            'ef_basic': data['LVEF (%)'].describe().to_dict(),
            'ef_distribution': {
                'values': hist_values.tolist(),
                'bins': hist_bins.tolist()
            },
            'ef_categories': {
                'reduced': (data['LVEF (%)'] < 40).sum(),
                'mid_range': ((data['LVEF (%)'] >= 40) & (data['LVEF (%)'] < 50)).sum(),
                'preserved': (data['LVEF (%)'] >= 50).sum()
            }
        }

        if 'Heart rate (bpm)' in data.columns:
            ef_stats['cardiac_output'] = {
                'basic': data['LVCO (L/min)'].describe().to_dict(),
                'cardiac_index': (
                    data['LVCO (L/min)'] / data['BSA']
                ).describe().to_dict() if 'BSA' in data.columns else None
            }

        return ef_stats

class RVAgent(BaseOrganAgent):
    """Right Ventricle Analysis Agent"""

    def __init__(self, api_key: str):
        phenotypes = [
            'RVEDV (mL)', 'RVESV (mL)', 'RVSV (mL)', 'RVEF (%)'
        ]
        super().__init__('RV', phenotypes, api_key)

        self.normal_ranges = {
            'RVEDV (mL)': (60, 180),
            'RVEF (%)': (47, 63)
        }

        self.system_prompt = """You are a specialized cardiac imaging expert focusing on Right Ventricular (RV) analysis. Your tasks include:

1. Volume Assessment:
   - Analyze EDV, ESV, and SV
   - Compare with LV volumes for ventricular interdependence
   - Identify RV dilation or underfilling

2. RV Function:
   - Evaluate RVEF and systolic function
   - Assess RV-LV interaction
   - Detect early dysfunction signs

3. Clinical Context:
   - Identify patterns suggesting pulmonary hypertension
   - Evaluate for RV failure signs
   - Assess ventricular interdependence

4. Specific Considerations:
   - Account for RV's complex geometry
   - Consider load-dependency of measurements
   - Evaluate regional vs global function

Guidelines:
- Use established RV-specific reference ranges
- Consider pressure/volume relationships
- Flag patterns suggesting pulmonary vascular disease
- Maintain awareness of technical limitations
- Consider impact of respiratory variation"""

        self.messages = [{"role": "system", "content": self.system_prompt}]

        self.rv_tools = {
            'volume_analysis': self._analyze_rv_volumes,
            'function_analysis': self._analyze_rv_function,
            'rv_lv_interaction': self._analyze_rv_lv_interaction
        }

    async def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Execute RV analysis pipeline"""
        self._validate_data(data)
        results = {}

        # Volume Analysis
        volume_results = self.rv_tools['volume_analysis'](data)
        results['volumes'] = volume_results

        # Function Analysis
        function_results = self.rv_tools['function_analysis'](data)
        results['function'] = function_results

        # RV-LV Interaction
        interaction_results = self.rv_tools['rv_lv_interaction'](data)
        results['rv_lv_interaction'] = interaction_results

        # Statistical Analysis
        rv_data = self.get_phenotype_data(data)
        corr_matrix, pvals = self.tools.correlation_analysis(rv_data)
        results['statistics'] = {
            'correlations': corr_matrix.to_dict(),
            'p_values': pvals.to_dict(),
            'pca': self.tools.pca_analysis(rv_data.dropna())
        }

        # GPT Analysis
        gpt_description = f"""
        RV Measurements Summary:
        - RVEDV: {data['RVEDV (mL)'].describe().to_dict()}
        - RVEF: {data['RVEF (%)'].describe().to_dict()}
        - Key Correlations: {results['statistics']['correlations']}
        """
        results['gpt_analysis'] = self.get_gpt_analysis(gpt_description)

        return results

    def _analyze_rv_volumes(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze RV volumes"""
        volume_data = data[['RVEDV (mL)', 'RVESV (mL)', 'RVSV (mL)']]
        volume_stats = self._calculate_basic_stats(volume_data)

        volume_stats.update({
            'volume_ratios': {
                'esv_to_edv': (data['RVESV (mL)'] / data['RVEDV (mL)']).describe().to_dict(),
                'sv_to_edv': (data['RVSV (mL)'] / data['RVEDV (mL)']).describe().to_dict()
            },
            'abnormal_volumes': {
                'dilated': (data['RVEDV (mL)'] > self.normal_ranges['RVEDV (mL)'][1]).sum(),
                'small': (data['RVEDV (mL)'] < self.normal_ranges['RVEDV (mL)'][0]).sum()
            }
        })

        return volume_stats

    def _analyze_rv_function(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze RV function"""
        hist_values, hist_bins = np.histogram(data['RVEF (%)'].dropna(), bins=20)

        function_stats = {
            'ef_stats': data['RVEF (%)'].describe().to_dict(),
            'ef_distribution': {
                'values': hist_values.tolist(),
                'bins': hist_bins.tolist()
            },
            'function_categories': {
                'reduced': (data['RVEF (%)'] < 45).sum(),
                'borderline': ((data['RVEF (%)'] >= 45) & (data['RVEF (%)'] < 47)).sum(),
                'normal': (data['RVEF (%)'] >= 47).sum()
            }
        }

        if 'Heart rate (bpm)' in data.columns:
            function_stats['rv_output'] = (
                data['RVSV (mL)'] * data['Heart rate (bpm)'] / 1000
            ).describe().to_dict()

        return function_stats

    def _analyze_rv_lv_interaction(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze RV-LV interaction"""
        if 'LVEDV (mL)' in data.columns and 'LVEF (%)' in data.columns:
            rv_lv_ratio = data['RVEDV (mL)'] / data['LVEDV (mL)']
            ef_correlation = pearsonr(
                data['RVEF (%)'].dropna(),
                data['LVEF (%)'].dropna()
            )

            return {
                'rv_lv_ratio': rv_lv_ratio.describe().to_dict(),
                'ef_correlation': {
                    'correlation': ef_correlation[0],
                    'p_value': ef_correlation[1]
                },
                'volume_interdependence': {
                    'correlation': pearsonr(
                        data['RVEDV (mL)'].dropna(),
                        data['LVEDV (mL)'].dropna()
                    )
                }
            }
        return {}

class LAAgent(BaseOrganAgent):
    """Left Atrial Analysis Agent"""

    def __init__(self, api_key: str):
        phenotypes = [
            'LAV max (mL)', 'LAV min (mL)', 'LASV (mL)', 'LAEF (%)'
        ]
        super().__init__('LA', phenotypes, api_key)

        self.normal_ranges = {
            'LAV max (mL)': (16, 64),
            'LAEF (%)': (50, 70)
        }

        self.system_prompt = """You are a cardiac imaging expert specialized in Left Atrial (LA) analysis. Your responsibilities:

1. Volume Analysis:
   - Maximum/minimum volumes
   - Reservoir function assessment
   - LA remodeling patterns

2. Phasic Function:
   - Reservoir function
   - Conduit function
   - Contractile function
   - LA-LV coupling

3. Clinical Implications:
   - Diastolic dysfunction markers
   - Atrial fibrillation risk
   - LA strain assessment

Technical Guidelines:
- Report indexed values when available
- Consider loading conditions
- Assess LA-LV interaction
- Flag subclinical dysfunction"""

        self.messages = [{"role": "system", "content": self.system_prompt}]

        self.la_tools = {
            'volume_analysis': self._analyze_la_volumes,
            'function_analysis': self._analyze_la_function,
            'la_lv_interaction': self._analyze_la_lv_interaction
        }

    async def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Execute LA analysis pipeline"""
        self._validate_data(data)
        results = {}

        # Volume analysis
        results['volumes'] = self.la_tools['volume_analysis'](data)

        # Function analysis
        results['function'] = self.la_tools['function_analysis'](data)

        # LA-LV interaction
        results['la_lv_interaction'] = self.la_tools['la_lv_interaction'](data)

        # Statistical analysis
        la_data = self.get_phenotype_data(data)
        corr_matrix, pvals = self.tools.correlation_analysis(la_data)
        results['statistics'] = {
            'correlations': corr_matrix.to_dict(),
            'p_values': pvals.to_dict(),
            'pca': self.tools.pca_analysis(la_data.dropna())
        }

        # GPT analysis
        description = f"""
        LA Measurements:
        - LAV max: {data['LAV max (mL)'].describe().to_dict()}
        - LAEF: {data['LAEF (%)'].describe().to_dict()}
        - Key findings: {results['statistics']['correlations']}
        """
        results['gpt_analysis'] = self.get_gpt_analysis(description)

        return results

    def _analyze_la_volumes(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze LA volumes"""
        volume_data = data[['LAV max (mL)', 'LAV min (mL)', 'LASV (mL)']]
        volume_stats = self._calculate_basic_stats(volume_data)

        # Additional volume metrics
        volume_stats.update({
            'volume_ratios': {
                'min_to_max': (data['LAV min (mL)'] / data['LAV max (mL)']).describe().to_dict(),
                'expansion_index': ((data['LAV max (mL)'] - data['LAV min (mL)']) / 
                                  data['LAV min (mL)'] * 100).describe().to_dict()
            },
            'abnormal_volumes': {
                'dilated': (data['LAV max (mL)'] > self.normal_ranges['LAV max (mL)'][1]).sum(),
                'small': (data['LAV max (mL)'] < self.normal_ranges['LAV max (mL)'][0]).sum()
            }
        })

        return volume_stats

    def _analyze_la_function(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze LA function"""
        hist_vals, hist_bins = np.histogram(data['LAEF (%)'].dropna(), bins=20)

        return {
            'ef_stats': data['LAEF (%)'].describe().to_dict(),
            'ef_distribution': {
                'values': hist_vals.tolist(),
                'bins': hist_bins.tolist()
            },
            'reservoir_function': (
                data['LAV max (mL)'] - data['LAV min (mL)']
            ).describe().to_dict()
        }

    def _analyze_la_lv_interaction(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze LA-LV interaction"""
        if 'LVEDV (mL)' in data.columns and 'LVEF (%)' in data.columns:
            la_lv_ratio = data['LAV max (mL)'] / data['LVEDV (mL)']

            laef_data = data['LAEF (%)'].dropna()
            lvef_data = data['LVEF (%)'].dropna()

            if len(laef_data) == len(lvef_data):
                ef_correlation = pearsonr(laef_data, lvef_data)
                return {
                    'la_lv_ratio': la_lv_ratio.describe().to_dict(),
                    'ef_correlation': {
                        'correlation': ef_correlation[0],
                        'p_value': ef_correlation[1]
                    }
                }
            else:
                return {
                    'la_lv_ratio': la_lv_ratio.describe().to_dict(),
                    'ef_correlation': 'Unable to calculate due to mismatched data lengths'
                }
        return {}

class RAAgent(BaseOrganAgent):
    """Right Atrial Analysis Agent"""

    def __init__(self, api_key: str):
        phenotypes = [
            'RAV max (mL)', 'RAV min (mL)', 'RASV (mL)', 'RAEF (%)'
        ]
        super().__init__('RA', phenotypes, api_key)

        self.normal_ranges = {
            'RAV max (mL)': (25, 58),
            'RAEF (%)': (46, 68)
        }

        self.system_prompt = """You are a cardiac imaging expert specialized in Right Atrial (RA) analysis. Core responsibilities:

1. Volume Analysis:
   - Max/min volumes and reservoir function
   - RA remodeling patterns
   - Volume response to loading conditions

2. Function Assessment:
   - Reservoir, conduit, booster pump functions
   - RA-RV coupling mechanics
   - Impact on RV filling

3. Clinical Integration:
   - Right heart dysfunction markers
   - Pulmonary hypertension indicators
   - Tricuspid valve function impact

4. Advanced Analysis:
   - Phasic function quantification
   - Loading condition effects
   - Pressure-volume relationships"""

        self.messages = [{"role": "system", "content": self.system_prompt}]

        self.ra_tools = {
            'volume_analysis': self._analyze_ra_volumes,
            'function_analysis': self._analyze_ra_function,
            'ra_rv_interaction': self._analyze_ra_rv_interaction
        }

    async def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Execute RA analysis pipeline"""
        self._validate_data(data)
        results = {}

        results['volumes'] = self.ra_tools['volume_analysis'](data)
        results['function'] = self.ra_tools['function_analysis'](data)
        results['ra_rv_interaction'] = self.ra_tools['ra_rv_interaction'](data)

        ra_data = self.get_phenotype_data(data)
        corr_matrix, pvals = self.tools.correlation_analysis(ra_data)
        results['statistics'] = {
            'correlations': corr_matrix.to_dict(),
            'p_values': pvals.to_dict(),
            'pca': self.tools.pca_analysis(ra_data.dropna())
        }

        description = f"""
        RA Measurements:
        - RAV max: {data['RAV max (mL)'].describe().to_dict()}
        - RAEF: {data['RAEF (%)'].describe().to_dict()}
        - Key findings: {results['statistics']['correlations']}
        """
        results['gpt_analysis'] = self.get_gpt_analysis(description)

        return results

    def _analyze_ra_volumes(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze RA volumes"""
        volume_data = data[['RAV max (mL)', 'RAV min (mL)', 'RASV (mL)']]
        volume_stats = self._calculate_basic_stats(volume_data)

        volume_stats.update({
            'volume_ratios': {
                'min_to_max': (data['RAV min (mL)'] / data['RAV max (mL)']).describe().to_dict(),
                'expansion_index': ((data['RAV max (mL)'] - data['RAV min (mL)']) / 
                                  data['RAV min (mL)'] * 100).describe().to_dict()
            },
            'abnormal_volumes': {
                'dilated': (data['RAV max (mL)'] > self.normal_ranges['RAV max (mL)'][1]).sum(),
                'small': (data['RAV max (mL)'] < self.normal_ranges['RAV max (mL)'][0]).sum()
            }
        })

        return volume_stats

    def _analyze_ra_function(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze RA function"""
        hist_vals, hist_bins = np.histogram(data['RAEF (%)'].dropna(), bins=20)

        return {
            'ef_stats': data['RAEF (%)'].describe().to_dict(),
            'ef_distribution': {
                'values': hist_vals.tolist(),
                'bins': hist_bins.tolist()
            },
            'reservoir_function': (
                data['RAV max (mL)'] - data['RAV min (mL)']
            ).describe().to_dict()
        }

    def _analyze_ra_rv_interaction(self, data: pd.DataFrame) -> Dict[str, Any]:
        if 'RVEDV (mL)' in data.columns and 'RVEF (%)' in data.columns:
            ra_rv_ratio = data['RAV max (mL)'] / data['RVEDV (mL)']

            # Ensure equal-length data before correlation
            raef_data = data['RAEF (%)'].dropna()
            rvef_data = data['RVEF (%)'].dropna()

            # Find common index
            common_index = raef_data.index.intersection(rvef_data.index)

            if not common_index.empty:
                ef_correlation = pearsonr(
                    raef_data.loc[common_index],
                    rvef_data.loc[common_index]
                )

                return {
                    'ra_rv_ratio': ra_rv_ratio.describe().to_dict(),
                    'ef_correlation': {
                        'correlation': ef_correlation[0],
                        'p_value': ef_correlation[1]
                    }
                }
        return {}

class AortaAgent(BaseOrganAgent):
    """Aorta Analysis Agent"""

    def __init__(self, api_key: str):
        phenotypes = [
            'AAo max area (mm2)', 'AAo min area (mm2)', 
            'AAo distensibility (10-3 mmHg-1)',
            'DAo max area (mm2)', 'DAo min area (mm2)', 
            'DAo distensibility (10-3 mmHg-1)'
        ]
        super().__init__('Aorta', phenotypes, api_key)

        self.normal_ranges = {
            'AAo distensibility (10-3 mmHg-1)': (2.0, 5.0),
            'DAo distensibility (10-3 mmHg-1)': (2.5, 5.5)
        }

        self.system_prompt = """You are an expert in aortic imaging analysis. Your tasks:

1. Area Analysis:
   - Ascending/Descending aortic dimensions
   - Regional differences
   - Area strain calculation

2. Distensibility Assessment:
   - Regional compliance patterns
   - Age-related changes
   - Pressure-strain relationships

3. Clinical Integration:
   - Arterial stiffness markers
   - Early disease detection
   - Vascular aging assessment

4. Technical Aspects:
   - Image quality validation
   - Measurement standardization
   - Regional variations analysis"""

        self.messages = [{"role": "system", "content": self.system_prompt}]

        self.aorta_tools = {
            'area_analysis': self._analyze_areas,
            'distensibility_analysis': self._analyze_distensibility,
            'aorta_cardiac_interaction': self._analyze_aorta_cardiac_interaction
        }

    async def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        results = {}

        results['areas'] = self.aorta_tools['area_analysis'](data)
        results['distensibility'] = self.aorta_tools['distensibility_analysis'](data)
        results['cardiac_interaction'] = self.aorta_tools['aorta_cardiac_interaction'](data)

        aorta_data = self.get_phenotype_data(data)
        corr_matrix, pvals = self.tools.correlation_analysis(aorta_data)
        results['statistics'] = {
            'correlations': corr_matrix.to_dict(),
            'p_values': pvals.to_dict(),
            'pca': self.tools.pca_analysis(aorta_data.dropna())
        }

        description = f"""
        Aortic Measurements:
        AAo Distensibility: {data['AAo distensibility (10-3 mmHg-1)'].describe().to_dict()}
        DAo Distensibility: {data['DAo distensibility (10-3 mmHg-1)'].describe().to_dict()}
        Key Findings: {results['statistics']['correlations']}
        """
        results['gpt_analysis'] = self.get_gpt_analysis(description)

        return results

    def _analyze_areas(self, data: pd.DataFrame) -> Dict[str, Any]:
        aao_data = data[['AAo max area (mm2)', 'AAo min area (mm2)']]
        dao_data = data[['DAo max area (mm2)', 'DAo min area (mm2)']]

        return {
            'aao_stats': {
                'basic': aao_data.describe().to_dict(),
                'area_change': (
                    (aao_data['AAo max area (mm2)'] - aao_data['AAo min area (mm2)']) /
                    aao_data['AAo min area (mm2)'] * 100
                ).describe().to_dict()
            },
            'dao_stats': {
                'basic': dao_data.describe().to_dict(),
                'area_change': (
                    (dao_data['DAo max area (mm2)'] - dao_data['DAo min area (mm2)']) /
                    dao_data['DAo min area (mm2)'] * 100
                ).describe().to_dict()
            }
        }

    def _analyze_distensibility(self, data: pd.DataFrame) -> Dict[str, Any]:
        return {
            'aao_distensibility': data['AAo distensibility (10-3 mmHg-1)'].describe().to_dict(),
            'dao_distensibility': data['DAo distensibility (10-3 mmHg-1)'].describe().to_dict(),
            'regional_comparison': (
                data['AAo distensibility (10-3 mmHg-1)'] /
                data['DAo distensibility (10-3 mmHg-1)']
            ).describe().to_dict(),
            'abnormal_patterns': {
                'stiff_aao': (
                    data['AAo distensibility (10-3 mmHg-1)'] < 
                    self.normal_ranges['AAo distensibility (10-3 mmHg-1)'][0]
                ).sum(),
                'stiff_dao': (
                    data['DAo distensibility (10-3 mmHg-1)'] < 
                    self.normal_ranges['DAo distensibility (10-3 mmHg-1)'][0]
                ).sum()
            }
        }

    def _analyze_aorta_cardiac_interaction(self, data: pd.DataFrame) -> Dict[str, Any]:
        if 'LVEF (%)' in data.columns:
            aao_data = data['AAo distensibility (10-3 mmHg-1)'].dropna()
            lv_data = data['LVEF (%)'].dropna()
            dao_data = data['DAo distensibility (10-3 mmHg-1)'].dropna()

            # Find common index for AAo-LV correlation
            aao_lv_common_index = aao_data.index.intersection(lv_data.index)

            # Find common index for DAo-LV correlation
            dao_lv_common_index = dao_data.index.intersection(lv_data.index)

            result = {}

            if not aao_lv_common_index.empty:
                aao_lv_corr = pearsonr(
                    aao_data.loc[aao_lv_common_index],
                    lv_data.loc[aao_lv_common_index]
                )
                result['aao_lv_correlation'] = {
                    'correlation': aao_lv_corr[0],
                    'p_value': aao_lv_corr[1]
                }

            if not dao_lv_common_index.empty:
                dao_lv_corr = pearsonr(
                    dao_data.loc[dao_lv_common_index],
                    lv_data.loc[dao_lv_common_index]
                )
                result['dao_lv_correlation'] = {
                    'correlation': dao_lv_corr[0],
                    'p_value': dao_lv_corr[1]
                }

            return result

        return {}

class StrainAgent(BaseOrganAgent):
    """Strain Analysis Agent"""

    def __init__(self, api_key: str):
        phenotypes = [
            'WT_Global (mm)', 'Ecc_Global (%)', 'Err_Global (%)', 'Ell_Global (%)'
        ]
        for i in range(1, 17):
            phenotypes.extend([
                f'WT_AHA_{i} (mm)',
                f'Ecc_AHA_{i} (%)',
                f'Err_AHA_{i} (%)'
            ])
        for i in range(1, 7):
            phenotypes.append(f'Ell_{i} (%)')

        super().__init__('Strain', phenotypes, api_key)

        self.system_prompt = """You are a myocardial strain imaging expert. Core responsibilities:

1. Global Strain:
   - Circumferential (Ecc)
   - Radial (Err)
   - Longitudinal (Ell)
   - Integration of multi-directional strain

2. Regional Analysis:
   - 17-segment AHA model interpretation
   - Regional dysfunction patterns
   - Wall thickness correlation

3. Clinical Applications:
   - Early dysfunction detection
   - Regional wall motion assessment
   - Dyssynchrony evaluation

4. Quality Assessment:
   - Strain curve analysis
   - Measurement reliability
   - Technical limitations"""

        self.messages = [{"role": "system", "content": self.system_prompt}]

        self.strain_tools = {
            'regional_analysis': self._analyze_regional_strain,
            'global_analysis': self._analyze_global_strain,
            'wall_thickness': self._analyze_wall_thickness
        }

    async def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        results = {}

        results['regional'] = self.strain_tools['regional_analysis'](data)
        results['global'] = self.strain_tools['global_analysis'](data)
        results['wall_thickness'] = self.strain_tools['wall_thickness'](data)

        strain_data = self.get_phenotype_data(data)
        corr_matrix, pvals = self.tools.correlation_analysis(strain_data)
        results['statistics'] = {
            'correlations': corr_matrix.to_dict(),
            'p_values': pvals.to_dict(),
            'pca': self.tools.pca_analysis(strain_data.dropna())
        }

        description = f"""
        Global Strain Values:
        - Ecc: {data['Ecc_Global (%)'].describe().to_dict()}
        - Err: {data['Err_Global (%)'].describe().to_dict()}
        - Ell: {data['Ell_Global (%)'].describe().to_dict()}
        Wall Thickness: {data['WT_Global (mm)'].describe().to_dict()}
        """
        results['gpt_analysis'] = self.get_gpt_analysis(description)

        return results

    def _analyze_regional_strain(self, data: pd.DataFrame) -> Dict[str, Any]:
        results = {}

        for i in range(1, 17):
            segment_data = {
                'wall_thickness': data[f'WT_AHA_{i} (mm)'].describe().to_dict(),
                'circumferential_strain': data[f'Ecc_AHA_{i} (%)'].describe().to_dict(),
                'radial_strain': data[f'Err_AHA_{i} (%)'].describe().to_dict()
            }
            results[f'segment_{i}'] = segment_data

        for i in range(1, 7):
            results[f'longitudinal_segment_{i}'] = {
                'strain': data[f'Ell_{i} (%)'].describe().to_dict()
            }

        # Calculate segment variability
        circ_std = data[[f'Ecc_AHA_{i} (%)' for i in range(1, 17)]].std(axis=1)
        rad_std = data[[f'Err_AHA_{i} (%)' for i in range(1, 17)]].std(axis=1)

        results['segmental_heterogeneity'] = {
            'circumferential_std': circ_std.describe().to_dict(),
            'radial_std': rad_std.describe().to_dict()
        }

        return results

    def _analyze_global_strain(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze global strain"""
        # Get histograms for each strain type
        hist_ecc_vals, hist_ecc_bins = np.histogram(data['Ecc_Global (%)'].dropna(), bins=20)
        hist_err_vals, hist_err_bins = np.histogram(data['Err_Global (%)'].dropna(), bins=20)
        hist_ell_vals, hist_ell_bins = np.histogram(data['Ell_Global (%)'].dropna(), bins=20)

        global_strain = {
            'circumferential': {
                'basic_stats': data['Ecc_Global (%)'].describe().to_dict(),
                'distribution': {
                    'values': hist_ecc_vals.tolist(),
                    'bins': hist_ecc_bins.tolist()
                }
            },
            'radial': {
                'basic_stats': data['Err_Global (%)'].describe().to_dict(), 
                'distribution': {
                    'values': hist_err_vals.tolist(),
                    'bins': hist_err_bins.tolist()
                }
            },
            'longitudinal': {
                'basic_stats': data['Ell_Global (%)'].describe().to_dict(),
                'distribution': {
                    'values': hist_ell_vals.tolist(),
                    'bins': hist_ell_bins.tolist()
                }
            }
        }

        # Strain ratios
        global_strain['strain_ratios'] = {
            'radial_to_circumferential': (
                data['Err_Global (%)'].abs() /
                data['Ecc_Global (%)'].abs()
            ).describe().to_dict(),
            'longitudinal_to_circumferential': (
                data['Ell_Global (%)'] /
                data['Ecc_Global (%)']
            ).describe().to_dict()
        }

        return global_strain

    def _analyze_wall_thickness(self, data: pd.DataFrame) -> Dict[str, Any]:
        wt_cols = [f'WT_AHA_{i} (mm)' for i in range(1, 17)]

        # Find common non-null indices for wall thickness and strain
        wt_data = data['WT_Global (mm)'].dropna()
        ecc_data = data['Ecc_Global (%)'].dropna()
        err_data = data['Err_Global (%)'].dropna()

        common_index = wt_data.index.intersection(ecc_data.index).intersection(err_data.index)

        thickness_strain_correlation = {
            'wt_ecc': pearsonr(
                wt_data.loc[common_index],
                ecc_data.loc[common_index]
            )[0] if not common_index.empty else None,
            'wt_err': pearsonr(
                wt_data.loc[common_index],
                err_data.loc[common_index]
            )[0] if not common_index.empty else None
        }

        return {
            'global': data['WT_Global (mm)'].describe().to_dict(),
            'segment_variation': {
                'std': data[wt_cols].std().to_dict(),
                'range': {
                    'min': data[wt_cols].min().to_dict(),
                    'max': data[wt_cols].max().to_dict()
                }
            },
            'thickness_strain_correlation': thickness_strain_correlation
        }

class ChiefAgent:
    """Chief Agent, responsible for coordinating all organ agents and integrating analysis results"""

    def __init__(self, api_key: str):
        # Initialize all agents
        self.agents = {
            'LV': LVAgent(api_key),
            'RV': RVAgent(api_key),
            'LA': LAAgent(api_key),
            'RA': RAAgent(api_key),
            'Aorta': AortaAgent(api_key),
            'Strain': StrainAgent(api_key)
        }

        self.memory = Memory()
        self.tools = StatisticalTools()
        self.latest_results = None
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)

        self.clinical_factors = [
            'Hypertension', 'Diabetes', 'Cardiac disease',
            'High cholesterol', 'Smoking status'
        ]

        self.system_prompt = """You are a senior cardiac imaging expert leading a team of specialized analysts. Your role:
1. Data Integration: Coordinate multi-chamber analysis and identify inter-organ relationships
2. Clinical Synthesis: Integrate findings and identify global dysfunction patterns
3. Research Applications: Identify key phenotypes and novel patterns
4. Quality Control: Validate measurements and standardize reporting
5. Pattern Recognition: Identify disease-specific cardiac remodeling patterns"""

        self.messages = [{"role": "system", "content": self.system_prompt}]

    def _get_organ_interaction(self, org1: str, org2: str, pair_data: pd.DataFrame) -> Dict[str, Any]:
        """Smart analysis of organ interactions"""
        try:
            # Filter columns for the two organs
            org1_cols = [col for col in pair_data.columns if org1 in col]
            org2_cols = [col for col in pair_data.columns if org2 in col]

            # Calculate interaction metrics
            interaction_metrics = {
                'correlation_matrix': pair_data[org1_cols + org2_cols].corr().to_dict(),
                'cross_organ_dependencies': {
                    'variance_ratio': np.var(pair_data[org1_cols].values) / 
                                    np.var(pair_data[org2_cols].values) if len(org2_cols) > 0 else None
                }
            }

            return interaction_metrics

        except Exception as e:
            logger.error(f"Organ interaction analysis error {org1} and {org2}: {e}")
            return {}

    def _identify_key_interaction_features(self, interaction_data):
        """Identify key interaction features"""
        # Use PCA or other dimensionality reduction to find most important interaction features
        pca = PCA(n_components=2)
        pca.fit(interaction_data)

        return {
            'top_interaction_components': pca.components_,
            'explained_variance_ratio': pca.explained_variance_ratio_
        }

    async def run_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:


        """Run full analysis pipeline"""
        try:
            results = {}

            # 1. Parallel organ analysis
            logger.info("Running organ-specific analyses...")
            organ_results = await self._run_parallel_analyses(data)
            results['organ_specific'] = organ_results

            # 2. Cross-organ analysis
            results['cross_organ'] = await self._analyze_cross_organ_relationships(data)

            # 3. Key phenotype mining
            results['phenotype_importance'] = self._discover_key_phenotypes(data)

            # 4. System-level analysis
            results['system_level'] = self._system_level_analysis(data)

            # 5. GPT overall analysis
            results['gpt_analysis'] = self._get_gpt_analysis(data, results)

            self.latest_results = results
            self.memory.store_analysis('full_analysis', results)

            return results

        except Exception as e:
            logger.error(f"Error in analysis pipeline: {e}")
            raise

    async def _run_parallel_analyses(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run organ analyses in parallel"""
        organ_results = {}
        tasks = []

        for name, agent in self.agents.items():
            tasks.append(agent.analyze(data))

        results = await asyncio.gather(*tasks)
        for name, result in zip(self.agents.keys(), results):
            organ_results[name] = result

        return organ_results

    async def _analyze_cross_organ_relationships(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze cross-organ relationships"""
        # Get all phenotypes
        all_phenotypes = []
        for agent in self.agents.values():
            all_phenotypes.extend(agent.phenotypes)

        # 1. Correlation analysis
        phenotype_data = data[all_phenotypes]
        corr_matrix, pvals = self.tools.correlation_analysis(phenotype_data)

        # 2. Organ pair analysis
        organ_pairs = [
            ('LV', 'RV'), ('LA', 'LV'), ('RA', 'RV'),
            ('LV', 'Aorta'), ('RV', 'Aorta')
        ]

        pair_correlations = await self._analyze_organ_pairs(data, organ_pairs)

        return {
            'correlations': corr_matrix.to_dict(),
            'p_values': pvals.to_dict(),
            'organ_pairs': pair_correlations,
            'pca': self.tools.pca_analysis(phenotype_data.dropna())
        }

    async def _analyze_organ_pairs(self, data: pd.DataFrame, 
                                 organ_pairs: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Analyze organ pair relationships"""
        results = {}
        for org1, org2 in organ_pairs:
            pheno1 = self.agents[org1].phenotypes
            pheno2 = self.agents[org2].phenotypes
            pair_data = data[pheno1 + pheno2].dropna()

            results[f"{org1}_{org2}"] = {
                'correlation': pair_data.corr().to_dict(),
                'interaction': self._get_organ_interaction(org1, org2, pair_data)
            }
        return results

    def _calculate_phenotype_score(self, data: pd.DataFrame, phenotype: str) -> Optional[float]:
        """Calculate phenotype score (0-1 range)"""
        try:
            y = data[phenotype]
            stats = y.describe()

            # 1. CV calculation and normalization
            cv = stats['std'] / abs(stats['mean']) if stats['mean'] != 0 else 0
            cv_norm = np.tanh(cv)  # Use tanh to compress to 0-1

            # 2. Outlier ratio
            outliers = self.tools.outlier_detection(y)
            outlier_ratio = outliers.mean()

            # 3. Range ratio calculation and normalization
            range_ratio = (stats['max'] - stats['min']) / abs(stats['max']) if stats['max'] != 0 else 0
            range_norm = np.tanh(range_ratio)

            # 4. Median deviation and normalization
            median_dev = abs(stats['50%'] - stats['mean']) / stats['std'] if stats['std'] != 0 else 0
            median_norm = np.tanh(median_dev)

            # Calculate weighted score
            score = (
                cv_norm * 0.4 +           # Variability
                (1 - outlier_ratio) * 0.3 +  # Stability
                range_norm * 0.2 +        # Range
                median_norm * 0.1         # Median deviation
            )

            return min(max(score, 0), 1)  # Ensure score is in 0-1 range

        except Exception as e:
            logger.warning(f"Error calculating score for phenotype {phenotype}: {e}")
            return None

    def _calculate_phenotype_rankings(self, sorted_phenotypes):
        """Calculate phenotype rankings"""
        return {
            'top_10': dict(sorted_phenotypes[:10]),
            'quartiles': {
                'Q1': dict(sorted_phenotypes[:int(len(sorted_phenotypes)*0.25)]),
                'Q2': dict(sorted_phenotypes[int(len(sorted_phenotypes)*0.25):int(len(sorted_phenotypes)*0.5)]),
                'Q3': dict(sorted_phenotypes[int(len(sorted_phenotypes)*0.5):int(len(sorted_phenotypes)*0.75)]),
                'Q4': dict(sorted_phenotypes[int(len(sorted_phenotypes)*0.75):])
            },
            'score_distribution': {
                'mean': np.mean([score for _, score in sorted_phenotypes]),
                'median': np.median([score for _, score in sorted_phenotypes]),
                'std': np.std([score for _, score in sorted_phenotypes])
            }
        }

    def _calculate_vascular_interaction(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate vascular-ventricular interaction"""
        if 'LVEF (%)' in data.columns and 'AAo distensibility (10-3 mmHg-1)' in data.columns:
            # Calculate correlation between LVEF and AAo distensibility
            lvef_data = data['LVEF (%)'].dropna()
            aao_dist_data = data['AAo distensibility (10-3 mmHg-1)'].dropna()

            # Find common index
            common_index = lvef_data.index.intersection(aao_dist_data.index)

            if not common_index.empty:
                correlation = pearsonr(
                    lvef_data.loc[common_index],
                    aao_dist_data.loc[common_index]
                )

                return {
                    'lvef_aao_correlation': {
                        'correlation': correlation[0],
                        'p_value': correlation[1]
                    }
                }

        return {}

    def _calculate_strain_ef_correlation(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate correlation between strain and ejection fraction"""
        strain_types = ['Ecc_Global (%)', 'Err_Global (%)', 'Ell_Global (%)']

        correlations = {}
        for strain_type in strain_types:
            if strain_type in data.columns and 'LVEF (%)' in data.columns:
                strain_data = data[strain_type].dropna()
                lvef_data = data['LVEF (%)'].dropna()

                # Find common index
                common_index = strain_data.index.intersection(lvef_data.index)

                if not common_index.empty:
                    correlation = pearsonr(
                        strain_data.loc[common_index],
                        lvef_data.loc[common_index]
                    )

                    correlations[strain_type] = {
                        'correlation': correlation[0],
                        'p_value': correlation[1]
                    }

        return correlations

    def _calculate_global_scores(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate global comprehensive scores"""
        global_scores = {}

        # Ventricular function score
        global_scores['ventricular_score'] = {
            'lv_ef': data['LVEF (%)'].mean(),
            'rv_ef': data['RVEF (%)'].mean(),
            'ef_consistency': abs(data['LVEF (%)'].mean() - data['RVEF (%)'].mean())
        }

        # Atrial function score
        global_scores['atrial_score'] = {
            'la_ef': data['LAEF (%)'].mean(),
            'ra_ef': data['RAEF (%)'].mean(),
            'ef_consistency': abs(data['LAEF (%)'].mean() - data['RAEF (%)'].mean())
        }

        # Strain score
        strain_types = ['Ecc_Global (%)', 'Err_Global (%)', 'Ell_Global (%)']
        global_scores['strain_score'] = {
            strain_type: data[strain_type].mean() for strain_type in strain_types
        }

        # Vascular compliance score
        global_scores['vascular_score'] = {
            'aao_distensibility': data['AAo distensibility (10-3 mmHg-1)'].mean(),
            'dao_distensibility': data['DAo distensibility (10-3 mmHg-1)'].mean()
        }

        return global_scores

    def _discover_key_phenotypes(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Discover key phenotypes"""
        all_phenotypes = []
        for agent in self.agents.values():
            all_phenotypes.extend(agent.phenotypes)

        phenotype_scores = {}
        for phenotype in all_phenotypes:
            score = self._calculate_phenotype_score(data, phenotype)
            if score is not None:
                phenotype_scores[phenotype] = score

        sorted_phenotypes = sorted(
            phenotype_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return {
            'scores': phenotype_scores,
            'top_phenotypes': sorted_phenotypes[:30],
            'rankings': self._calculate_phenotype_rankings(sorted_phenotypes)
        }

    def _system_level_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """System-level analysis"""
        return {
            'cardiac_function': self._analyze_cardiac_function(data),
            'vascular_coupling': self._analyze_vascular_coupling(data),
            'strain_integration': self._analyze_strain_integration(data),
            'global_scores': self._calculate_global_scores(data)
        }

    def _analyze_cardiac_function(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze overall cardiac function"""
        return {
            'ventricular': {
                'ef_ratio': (data['LVEF (%)'] / data['RVEF (%)']).describe().to_dict(),
                'volume_ratio': (data['LVEDV (mL)'] / data['RVEDV (mL)']).describe().to_dict()
            },
            'atrial': {
                'ef_ratio': (data['LAEF (%)'] / data['RAEF (%)']).describe().to_dict(),
                'volume_ratio': (data['LAV max (mL)'] / data['RAV max (mL)']).describe().to_dict()
            }
        }

    def _analyze_vascular_coupling(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze vascular coupling"""
        return {
            'aortic_compliance': {
                'ascending': data['AAo distensibility (10-3 mmHg-1)'].describe().to_dict(),
                'descending': data['DAo distensibility (10-3 mmHg-1)'].describe().to_dict()
            },
            'ventricular_interaction': self._calculate_vascular_interaction(data)
        }

    def _analyze_strain_integration(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Integrate strain analysis"""
        return {
            'global_strain': {
                'circumferential': data['Ecc_Global (%)'].describe().to_dict(),
                'radial': data['Err_Global (%)'].describe().to_dict(),
                'longitudinal': data['Ell_Global (%)'].describe().to_dict()
            },
            'ef_correlation': self._calculate_strain_ef_correlation(data)
        }

    def _get_gpt_analysis(self, data: pd.DataFrame, 
                              results: Dict[str, Any]) -> str:
        """Get overall GPT analysis"""
        summary = f"""
        Analysis Data Summary:
        1. Ventricular Function: LV EF = {data['LVEF (%)'].mean():.1f}%, RV EF = {data['RVEF (%)'].mean():.1f}%
        2. Atrial Function: LA EF = {data['LAEF (%)'].mean():.1f}%, RA EF = {data['RAEF (%)'].mean():.1f}%
        3. Aortic Compliance: AAo = {data['AAo distensibility (10-3 mmHg-1)'].mean():.2f}
        4. Key Findings: {results['phenotype_importance']['top_phenotypes'][:5]}
        """

        # response =  client.chat.completions.create(model="gpt-4o-mini-2024-07-18",
        # messages=[
        #     {"role": "system", "content": self.system_prompt},
        #     {"role": "user", "content": f"分析以下心脏MRI数据:\n{summary}"}
        # ],
        # temperature=0,
        # api_key=self.api_key)

        response = self.client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Analyze the following cardiac MRI data:\n{summary}"}
                ]
        )

        return response.choices[0].message.content

