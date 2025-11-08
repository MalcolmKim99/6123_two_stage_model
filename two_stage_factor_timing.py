import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
# warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE


# 1: DATA LOADING AND PREPARATION

def load_and_prepare_data(filepath):
    """
    Load the 6-factor data and prepare it for analysis
    
    Parameters:
    -----------
    filepath : str
        Path to the 6_factor.xls file
        
    Returns:
    --------
    pd.DataFrame : Prepared dataframe with datetime index
    """
    # Read the data
    df = pd.read_csv(filepath)
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    
    # Sort by date
    df = df.sort_index()
    
    print(f"Data loaded: {len(df)} months from {df.index[0]} to {df.index[-1]}")
    print(f"Factors: {[col for col in df.columns if col != 'RF']}")
    
    return df


def calculate_market_drawdown(returns, window=3):
    """
    Calculate rolling drawdown for market risk regime identification
    
    Parameters:
    -----------
    returns : pd.Series
        Market returns (e.g., Mkt-RF or average of factors)
    window : int
        Rolling window in months for drawdown calculation
        
    Returns:
    --------
    pd.Series : Drawdown series
    """
    # Calculate cumulative returns
    cumulative = (1 + returns / 100).cumprod()
    
    # Calculate running maximum
    running_max = cumulative.rolling(window=window, min_periods=1).max()
    
    # Calculate drawdown
    drawdown = (cumulative - running_max) / running_max
    
    return drawdown


def construct_financial_turbulence(df, lookback=60):
    """
    Calculate financial turbulence measure following Kritzman & Li (2010)
    
    Formula: d_t = (y_t - μ)' Σ^(-1) (y_t - μ)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Factor returns dataframe
    lookback : int
        Lookback period for calculating mean and covariance
        
    Returns:
    --------
    pd.Series : Financial turbulence series
    """
    factor_cols = [col for col in df.columns if col != 'RF']
    factor_returns = df[factor_cols].values
    
    turbulence = []
    dates = []
    
    for i in range(lookback, len(df)):
        # Historical window
        hist_returns = factor_returns[i-lookback:i]
        
        # Current returns
        current_returns = factor_returns[i]
        
        # Calculate mean and covariance
        mu = hist_returns.mean(axis=0)
        cov = np.cov(hist_returns.T)
        
        # Add small value to diagonal for numerical stability
        cov += np.eye(len(factor_cols)) * 1e-6
        
        # Calculate Mahalanobis distance
        try:
            diff = current_returns - mu
            turb = diff @ np.linalg.inv(cov) @ diff.T
            turbulence.append(turb)
            dates.append(df.index[i])
        except:
            turbulence.append(np.nan)
            dates.append(df.index[i])
    
    return pd.Series(turbulence, index=dates, name='Financial_Turbulence')



# 2: STAGE ONE - MARKET RISK REGIME IDENTIFICATION

class MarketRegimeClassifier:
    """
    Stage 1: Identify market risk regimes using clustering and classification
    """
    
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.kmeans = None
        self.scaler = StandardScaler()
        self.classifier = None
        self.regime_labels = None

    '''
    def identify_regimes_clustering(self, drawdown_series):
        """
        Use K-Means clustering to identify market regimes based on drawdown
        
        Parameters:
        -----------
        drawdown_series : pd.Series
            Market drawdown series
            
        Returns:
        --------
        pd.Series : Regime labels (0=normal, 1=correction, 2=bear market)
        """
        # Reshape for sklearn
        X = drawdown_series.values.reshape(-1, 1)
        
        # Fit K-Means
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        clusters = self.kmeans.fit_predict(X)
        
        # Sort clusters by mean drawdown (0=normal, 1=correction, 2=bear)
        cluster_means = []
        for i in range(self.n_clusters):
            cluster_means.append(drawdown_series[clusters == i].mean())
        
        # Create mapping: most negative = 2 (bear), least negative = 0 (normal)
        sorted_clusters = np.argsort(cluster_means)
        mapping = {sorted_clusters[i]: i for i in range(self.n_clusters)}
        
        # Apply mapping
        regime_labels = pd.Series([mapping[c] for c in clusters], 
                                  index=drawdown_series.index,
                                  name='Market_Regime')
        
        self.regime_labels = regime_labels
        
        print("\nMarket Regime Distribution:")
        print(regime_labels.value_counts().sort_index())
        print("\nRegime Statistics:")
        for regime in range(self.n_clusters):
            regime_data = drawdown_series[regime_labels == regime]
            print(f"  Regime {regime}: Mean DD = {regime_data.mean():.3f}, "
                  f"Count = {len(regime_data)} ({len(regime_data)/len(drawdown_series)*100:.1f}%)")
        
        return regime_labels
    '''
    def identify_regimes_clustering(self, drawdown_series):
        """
        Use K-Means clustering to identify market regimes based on drawdown
        
        Parameters:
        -----------
        drawdown_series : pd.Series
            Market drawdown series
            
        Returns:
        --------
        pd.Series : Regime labels (0=normal, 1=correction, 2=bear market)
        """
        # Reshape for sklearn
        X = drawdown_series.values.reshape(-1, 1)
        
        # Fit K-Means
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        clusters = self.kmeans.fit_predict(X)
        
        # Sort clusters by mean drawdown (ascending = most negative first)
        cluster_means = []
        for i in range(self.n_clusters):
            cluster_means.append(drawdown_series[clusters == i].mean())
        
        # Create mapping:
        # sorted_clusters[0] = cluster with most negative mean (bear) -> map to 2
        # sorted_clusters[1] = cluster with middle mean (correction) -> map to 1
        # sorted_clusters[2] = cluster with least negative mean (normal) -> map to 0
        sorted_clusters = np.argsort(cluster_means)  # Ascending order
        mapping = {
            sorted_clusters[0]: 2,  # Most negative -> Bear Market (Regime 2)
            sorted_clusters[1]: 1,  # Middle -> Correction (Regime 1)
            sorted_clusters[2]: 0   # Least negative -> Normal (Regime 0)
        }
        
        # Apply mapping
        regime_labels = pd.Series([mapping[c] for c in clusters], 
                                  index=drawdown_series.index,
                                  name='Market_Regime')
        
        self.regime_labels = regime_labels
        
        print("\nMarket Regime Distribution:")
        print(regime_labels.value_counts().sort_index())
        print("\nRegime Statistics:")
        for regime in range(self.n_clusters):
            regime_data = drawdown_series[regime_labels == regime]
            regime_name = ['Normal', 'Correction', 'Bear Market'][regime]
            print(f"  Regime {regime} ({regime_name}): Mean DD = {regime_data.mean():.3f}, "
                  f"Count = {len(regime_data)} ({len(regime_data)/len(drawdown_series)*100:.1f}%)")
        
        return regime_labels
    
        
    
    def find_optimal_clusters(self, drawdown_series, max_k=9):
        """
        Use elbow method to find optimal number of clusters
        
        Parameters:
        -----------
        drawdown_series : pd.Series
            Market drawdown series
        max_k : int
            Maximum number of clusters to test
            
        Returns:
        --------
        matplotlib.figure.Figure : Elbow plot
        """
        X = drawdown_series.values.reshape(-1, 1)
        distortions = []
        K = range(1, max_k+1)
        
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            distortions.append(kmeans.inertia_)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(K, distortions, 'bo-')
        ax.set_xlabel('Number of Clusters')
        ax.set_ylabel('Distortion')
        ax.set_title('The Elbow Method Showing the Optimal K')
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def train_regime_predictor(self, X_features, y_regime, use_smote=True):
        """
        Train stacking ensemble classifier to predict market regimes
        
        Parameters:
        -----------
        X_features : pd.DataFrame
            Feature matrix (financial turbulence, macro vars, etc.)
        y_regime : pd.Series
            Regime labels from clustering
        use_smote : bool
            Whether to use SMOTE for class balancing
            
        Returns:
        --------
        dict : Training results
        """
        # Handle missing values
        X = X_features.ffill().bfill()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply SMOTE if requested
        if use_smote and len(np.unique(y_regime)) > 1:
            smote = SMOTE(random_state=42)
            X_scaled, y_regime = smote.fit_resample(X_scaled, y_regime) # type: ignore
            print(f"\nAfter SMOTE - Class distribution:")
            unique, counts = np.unique(y_regime, return_counts=True)
            for u, c in zip(unique, counts):
                print(f"  Regime {u}: {c} samples")
        
        # Build stacking ensemble
        base_classifiers = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)),
            ('svc', SVC(probability=True, random_state=42)),
            ('gnb', GaussianNB())
        ]
        
        # Train base classifiers and get predictions
        base_predictions = []
        for name, clf in base_classifiers:
            clf.fit(X_scaled, y_regime)
            base_predictions.append(clf.predict_proba(X_scaled))
        
        # Stack predictions
        X_stack = np.hstack(base_predictions)
        
        # Train meta-classifier (Logistic Regression)
        self.classifier = LogisticRegression(random_state=42, max_iter=1000)
        self.classifier.fit(X_stack, y_regime)
        
        # Store base classifiers
        self.base_classifiers = dict(base_classifiers)
        
        # Cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(self.classifier, X_stack, y_regime, 
                                     cv=tscv, scoring='accuracy')
        
        results = {
            'cv_accuracy_mean': cv_scores.mean(),
            'cv_accuracy_std': cv_scores.std(),
            'train_accuracy': accuracy_score(y_regime, self.classifier.predict(X_stack))
        }
        
        print(f"\nStage 1 Model Performance:")
        print(f"  Cross-validation accuracy: {results['cv_accuracy_mean']:.3f} (+/- {results['cv_accuracy_std']:.3f})")
        print(f"  Training accuracy: {results['train_accuracy']:.3f}")
        
        return results
    
    def predict_regime(self, X_features):
        """
        Predict market regime for new data
        
        Parameters:
        -----------
        X_features : pd.DataFrame
            Feature matrix
            
        Returns:
        --------
        np.array : Predicted regime labels
        """
        X = X_features.ffill().bfill()
        X_scaled = self.scaler.transform(X)
        
        # Get base classifier predictions
        base_predictions = []
        for name, clf in self.base_classifiers.items():
            base_predictions.append(clf.predict_proba(X_scaled))
        
        X_stack = np.hstack(base_predictions)
        
        return self.classifier.predict(X_stack) # type: ignore


# 3: STAGE TWO - FACTOR PERFORMANCE PREDICTION

class FactorPerformancePredictor:
    """
    Stage 2: Predict winning factor within each market regime
    """
    
    def __init__(self, factor_names):
        self.factor_names = factor_names
        self.models = {}  # One model per regime
        self.scalers = {}  # One scaler per regime
        
    def create_factor_labels(self, factor_returns):
        """
        Create binary labels for each factor (1 if best performing, 0 otherwise)
        
        Parameters:
        -----------
        factor_returns : pd.DataFrame
            Factor returns dataframe
            
        Returns:
        --------
        pd.DataFrame : Binary labels for each factor
        """
        labels = pd.DataFrame(index=factor_returns.index)
        
        for factor in self.factor_names:
            # Find which factor had the highest return each month
            best_factor = factor_returns[self.factor_names].idxmax(axis=1)
            labels[factor] = (best_factor == factor).astype(int)
        
        return labels
    
    def train_regime_specific_models(self, X_features, factor_returns, regimes):
        """
        Train separate models for each market regime
        
        Parameters:
        -----------
        X_features : pd.DataFrame
            Feature matrix
        factor_returns : pd.DataFrame
            Factor returns
        regimes : pd.Series
            Market regime labels
            
        Returns:
        --------
        dict : Training results by regime
        """
        # Create factor labels
        factor_labels = self.create_factor_labels(factor_returns)
        
        results = {}
        
        for regime in sorted(regimes.unique()):
            print(f"\n Training models for Regime {regime}")
            
            # Filter data for this regime
            regime_mask = regimes == regime
            X_regime = X_features[regime_mask]
            y_regime_labels = factor_labels[regime_mask]
            
            print(f"Regime {regime} data points: {len(X_regime)}")
            
            # Train a model for each factor
            regime_models = {}
            regime_results = {}
            
            for factor in self.factor_names:
                y = y_regime_labels[factor].values
                
                # Skip if no positive samples
                if y.sum() == 0:
                    print(f"  {factor}: No positive samples, skipping")
                    continue
                
                # Handle missing values
                X_clean = X_regime.ffill().bfill()
                
                # Scale features
                if regime not in self.scalers:
                    self.scalers[regime] = StandardScaler()
                    X_scaled = self.scalers[regime].fit_transform(X_clean)
                else:
                    X_scaled = self.scalers[regime].transform(X_clean)
                
                # Train Random Forest
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    class_weight='balanced'
                )
                model.fit(X_scaled, y)
                
                # Evaluate
                train_acc = accuracy_score(y, model.predict(X_scaled))
                
                regime_models[factor] = model
                regime_results[factor] = {
                    'train_accuracy': train_acc,
                    'positive_samples': y.sum(),
                    'negative_samples': len(y) - y.sum()
                }
                
                print(f"  {factor}: Train Acc = {train_acc:.3f}, "
                      f"Pos/Neg = {y.sum()}/{len(y)-y.sum()}")
            
            self.models[regime] = regime_models
            results[regime] = regime_results
        
        return results
    
    def predict_factor_probabilities(self, X_features, regime):
        """
        Predict probability of each factor outperforming in given regime
        
        Parameters:
        -----------
        X_features : pd.DataFrame or np.array
            Feature matrix (single time point or multiple)
        regime : int or array
            Market regime(s)
            
        Returns:
        --------
        pd.DataFrame : Probability predictions for each factor
        """
        # Ensure X is DataFrame
        if isinstance(X_features, pd.Series):
            X_features = X_features.to_frame().T
        
        # Handle missing values
        X_clean = X_features.ffill().bfill()
        
        # Handle single regime vs multiple regimes
        if isinstance(regime, (int, np.integer)):
            regimes = [regime] * len(X_clean)
        else:
            regimes = regime
        
        # Initialize results
        predictions = pd.DataFrame(
            index=X_clean.index if hasattr(X_clean, 'index') else range(len(X_clean)),
            columns=self.factor_names
        )
        
        for i, reg in enumerate(regimes):
            if reg not in self.models:
                # If regime model doesn't exist, use equal weights
                predictions.iloc[i] = 1.0 / len(self.factor_names)
                continue
            
            X_single = X_clean.iloc[[i]] if hasattr(X_clean, 'iloc') else X_clean[i:i+1]
            
            # Scale
            X_scaled = self.scalers[reg].transform(X_single)
            
            # Predict for each factor
            for factor in self.factor_names:
                if factor in self.models[reg]:
                    prob = self.models[reg][factor].predict_proba(X_scaled)[0, 1]
                    predictions.iloc[i][factor] = prob
                else:
                    predictions.iloc[i][factor] = 0.0
            
            # Normalize to sum to 1
            row_sum = predictions.iloc[i].sum()
            if row_sum > 0:
                predictions.iloc[i] = predictions.iloc[i] / row_sum
            else:
                predictions.iloc[i] = 1.0 / len(self.factor_names)
        
        return predictions.astype(float)
    
    def get_feature_importance(self, regime, factor, top_n=10):
        """
        Get feature importance for a specific regime and factor
        
        Parameters:
        -----------
        regime : int
            Market regime
        factor : str
            Factor name
        top_n : int
            Number of top features to return
            
        Returns:
        --------
        pd.Series : Feature importances
        """
        if regime not in self.models or factor not in self.models[regime]:
            return None
        
        model = self.models[regime][factor]
        importances = pd.Series(
            model.feature_importances_,
            index=range(len(model.feature_importances_))
        )
        
        return importances.nlargest(top_n)


# 4: PORTFOLIO SIMULATION AND BACKTESTING

class FactorTimingStrategy:
    """
    Implement the complete two-stage factor timing strategy
    """
    
    def __init__(self, regime_classifier, factor_predictor):
        self.regime_classifier = regime_classifier
        self.factor_predictor = factor_predictor
        
    def backtest(self, factor_returns, X_features, train_end_idx, 
                 rebalance_freq='M', initial_capital=1.0):
        """
        Run rolling window backtest
        
        Parameters:
        -----------
        factor_returns : pd.DataFrame
            Factor returns
        X_features : pd.DataFrame
            Feature matrix
        train_end_idx : int
            Index where training period ends
        rebalance_freq : str
            Rebalancing frequency ('M' for monthly)
        initial_capital : float
            Initial capital
            
        Returns:
        --------
        dict : Backtest results
        """
        # Split data
        test_start = factor_returns.index[train_end_idx]
        test_returns = factor_returns.loc[test_start:]
        test_features = X_features.loc[test_start:]
        
        # Initialize portfolio
        portfolio_value = [initial_capital]
        portfolio_returns = []
        weights_history = []
        regime_history = []
        
        factor_cols = self.factor_predictor.factor_names
        
        print(f"\nBacktesting from {test_returns.index[0]} to {test_returns.index[-1]}")
        print(f"Test period: {len(test_returns)} months")
        
        for i in range(len(test_returns)):
            current_date = test_returns.index[i]
            
            # Get features for current period
            current_features = test_features.iloc[[i]]
            
            # Predict regime
            predicted_regime = self.regime_classifier.predict_regime(current_features)[0]
            regime_history.append(predicted_regime)
            
            # Predict factor probabilities
            factor_probs = self.factor_predictor.predict_factor_probabilities(
                current_features, predicted_regime
            )
            
            weights = factor_probs.iloc[0].values
            weights_history.append(weights)
            
            # Calculate portfolio return
            factor_ret = test_returns[factor_cols].iloc[i].values
            portfolio_ret = np.sum(weights * factor_ret)
            portfolio_returns.append(portfolio_ret)
            
            # Update portfolio value
            new_value = portfolio_value[-1] * (1 + portfolio_ret / 100)
            portfolio_value.append(new_value)
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'Date': test_returns.index,
            'Portfolio_Return': portfolio_returns,
            'Portfolio_Value': portfolio_value[1:],
            'Regime': regime_history
        })
        
        # Add weights
        for i, factor in enumerate(factor_cols):
            results_df[f'Weight_{factor}'] = [w[i] for w in weights_history]
        
        # Calculate metrics
        returns_series = pd.Series(portfolio_returns, index=test_returns.index)
        
        metrics = {
            'total_return': (portfolio_value[-1] / initial_capital - 1) * 100,
            'annualized_return': (portfolio_value[-1] / initial_capital) ** (12 / len(test_returns)) - 1,
            'annualized_volatility': returns_series.std() * np.sqrt(12),
            'sharpe_ratio': (returns_series.mean() * 12) / (returns_series.std() * np.sqrt(12)),
            'max_drawdown': self._calculate_max_drawdown(portfolio_value[1:]),
            'win_rate': (returns_series > 0).sum() / len(returns_series)
        }
        
        return {
            'results_df': results_df,
            'metrics': metrics,
            'portfolio_value': portfolio_value[1:],
            'returns': portfolio_returns
        }
    
    def _calculate_max_drawdown(self, portfolio_values):
        """Calculate maximum drawdown"""
        cummax = pd.Series(portfolio_values).cummax()
        drawdown = (pd.Series(portfolio_values) - cummax) / cummax
        return drawdown.min()
    
    def compare_strategies(self, factor_returns, equal_weight=True, 
                          train_end_idx=None):
        """
        Compare ML strategy with benchmarks
        
        Parameters:
        -----------
        factor_returns : pd.DataFrame
            Factor returns
        equal_weight : bool
            Whether to include equal-weight benchmark
        train_end_idx : int
            Training period end index
            
        Returns:
        --------
        pd.DataFrame : Comparison results
        """
        factor_cols = self.factor_predictor.factor_names
        
        # Get test period
        test_start = factor_returns.index[train_end_idx]
        test_returns = factor_returns.loc[test_start:, factor_cols]
        
        strategies = {}
        
        # Equal-weighted strategy
        if equal_weight:
            ew_returns = test_returns.mean(axis=1)
            ew_value = (1 + ew_returns / 100).cumprod()
            strategies['Equal_Weight'] = {
                'returns': ew_returns,
                'values': ew_value
            }
        
        # Add ML strategy (assume already run)
        # This would be filled in by the backtest results
        
        return strategies


# 5: VISUALIZATION AND REPORTING

def plot_regime_identification(drawdown, regimes, save_path=None):
    """
    Plot market drawdown with regime labels
    
    Parameters:
    -----------
    drawdown : pd.Series
        Drawdown series
    regimes : pd.Series
        Regime labels
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Color map for regimes
    colors = {0: 'green', 1: 'orange', 2: 'red'}
    labels = {0: 'Normal', 1: 'Correction', 2: 'Bear Market'}
    
    # Plot drawdown
    ax.plot(drawdown.index, drawdown.values, color='navy', linewidth=1.5, 
            label='Market Drawdown', zorder=1)
    
    # Highlight regimes
    for regime in sorted(regimes.unique()):
        regime_mask = regimes == regime
        ax.scatter(drawdown.index[regime_mask], drawdown.values[regime_mask],
                  c=colors[regime], label=labels[regime], alpha=0.6, s=20, zorder=2)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Drawdown', fontsize=12)
    ax.set_title('Market Drawdown and Risk Regimes', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_factor_probabilities(factor_probs, save_path=None):
    """
    Plot factor winning probabilities over time
    
    Parameters:
    -----------
    factor_probs : pd.DataFrame
        Factor probability predictions over time
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Stack area plot
    factor_probs.plot(kind='area', stacked=True, ax=ax, alpha=0.7)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title('Factor Outperformance Probabilities Over Time', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_ylim([0,1]) # type: ignore
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_portfolio_performance(results_dict, benchmark_returns=None, save_path=None):
    """
    Plot portfolio performance comparison
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary containing results from backtest
    benchmark_returns : pd.Series, optional
        Benchmark returns for comparison
    save_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    results_df = results_dict['results_df']
    
    # Plot 1: Cumulative returns
    ax1 = axes[0]
    ax1.plot(results_df['Date'], results_df['Portfolio_Value'], 
            linewidth=2, label='ML Strategy')
    
    if benchmark_returns is not None:
        bench_cumulative = (1 + benchmark_returns / 100).cumprod()
        ax1.plot(benchmark_returns.index, bench_cumulative,
                linewidth=2, label='Benchmark', linestyle='--')
    
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Portfolio Value', fontsize=12)
    ax1.set_title('Cumulative Portfolio Performance', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Monthly returns
    ax2 = axes[1]
    colors = ['green' if x > 0 else 'red' for x in results_df['Portfolio_Return']]
    ax2.bar(results_df['Date'], results_df['Portfolio_Return'], 
           color=colors, alpha=0.6)
    
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Monthly Return (%)', fontsize=12)
    ax2.set_title('Monthly Returns', fontsize=14, fontweight='bold')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def generate_performance_report(results_dict, strategy_name='ML Strategy'):
    """
    Generate a comprehensive performance report
    
    Parameters:
    -----------
    results_dict : dict
        Results from backtest
    strategy_name : str
        Name of the strategy
        
    Returns:
    --------
    str : Formatted report
    """
    metrics = results_dict['metrics']
    
    report = f"""
                    PERFORMANCE REPORT: {strategy_name}

Return Metrics:
--------------
  Total Return:              {metrics['total_return']:>8.2f}%
  Annualized Return:         {metrics['annualized_return']*100:>8.2f}%
  Annualized Volatility:     {metrics['annualized_volatility']:>8.2f}%

Risk-Adjusted Metrics:
---------------------
  Sharpe Ratio:              {metrics['sharpe_ratio']:>8.2f}
  Maximum Drawdown:          {metrics['max_drawdown']*100:>8.2f}%
  Win Rate:                  {metrics['win_rate']*100:>8.2f}%
    """
    
    return report


# 6: MAIN EXECUTION PIPELINE

def main_pipeline(data_filepath, output_dir='./outputs'):
    """
    Execute the complete two-stage factor timing pipeline
    
    Parameters:
    -----------
    data_filepath : str
        Path to 6_factor data file
    output_dir : str
        Directory to save outputs
        
    Returns:
    --------
    dict : Complete results
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("TWO-STAGE EQUITY FACTOR TIMING MODEL")
    

    # Step 1: Load Data

    print("\n[Step 1/7] Loading and preparing data")
    df = load_and_prepare_data(data_filepath)
    
    # Define factor columns (exclude RF)
    factor_cols = [col for col in df.columns if col != 'RF']
    print(f"Factor columns: {factor_cols}")
    

    # Step 2: Calculate Market Features

    print("\n[Step 2/7] Calculating market features")
    
    # Calculate market return (average of factors)
    df['Market_Return'] = df[factor_cols].mean(axis=1)
    
    # Calculate drawdown
    df['Drawdown'] = calculate_market_drawdown(df['Market_Return'])
    
    # Calculate financial turbulence
    turbulence = construct_financial_turbulence(df, lookback=60)
    df = df.join(turbulence)
    
    # Drop NaN values
    df = df.dropna()
    
    print(f"Data shape after feature engineering: {df.shape}")
    

    # Step 3: Stage 1 - Identify Market Regimes

    print("\n[Step 3/7] Stage 1: Identifying market risk regimes")
    
    regime_classifier = MarketRegimeClassifier(n_clusters=3)
    
    # Visualize elbow method
    print("  Finding optimal number of clusters")
    elbow_fig = regime_classifier.find_optimal_clusters(df['Drawdown'])
    elbow_fig.savefig(f'{output_dir}/elbow_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Identify regimes using clustering
    regimes = regime_classifier.identify_regimes_clustering(df['Drawdown'])
    df['Regime'] = regimes
    
    # Visualize regimes
    regime_fig = plot_regime_identification(df['Drawdown'], regimes)
    regime_fig.savefig(f'{output_dir}/market_regimes.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Train regime predictor
    print("\n  Training regime classification model")
    
    # Features for regime prediction (you can add more macro features here)
    regime_features = df[['Financial_Turbulence', 'Drawdown']]
    
    regime_results = regime_classifier.train_regime_predictor(
        regime_features, 
        regimes,
        use_smote=True
    )
    

    # Step 4: Stage 2 - Train Factor Predictors

    print("\n[Step 4/7] Stage 2: Training factor performance predictors")
    
    factor_predictor = FactorPerformancePredictor(factor_cols)
    
    # Features for factor prediction (you can add more features here)
    factor_features = df[['Financial_Turbulence', 'Drawdown']]
    
    # Train regime-specific models
    factor_results = factor_predictor.train_regime_specific_models(
        factor_features,
        df[factor_cols],
        df['Regime']
    )
    

    # Step 5: Backtest the Strategy

    print("\n[Step 5/7] Running backtest")
    
    # Define training period (e.g., first 60% of data)
    train_size = int(len(df) * 0.6)
    print(f"  Training period: {df.index[0]} to {df.index[train_size-1]}")
    print(f"  Test period: {df.index[train_size]} to {df.index[-1]}")
    
    strategy = FactorTimingStrategy(regime_classifier, factor_predictor)
    
    backtest_results = strategy.backtest(
        df[factor_cols + ['RF']],
        factor_features,
        train_size
    )
    

    # Step 6: Generate Visualizations

    print("\n[Step 6/7] Generating visualizations")
    
    # Portfolio performance
    perf_fig = plot_portfolio_performance(backtest_results)
    perf_fig.savefig(f'{output_dir}/portfolio_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Factor probabilities (for test period)
    test_dates = backtest_results['results_df']['Date']
    test_weights = backtest_results['results_df'][[f'Weight_{f}' for f in factor_cols]]
    test_weights.columns = factor_cols
    test_weights.index = test_dates
    
    prob_fig = plot_factor_probabilities(test_weights)
    prob_fig.savefig(f'{output_dir}/factor_probabilities.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    
    # Step 7: Generate Report

    print("\n[Step 7/7] Generating performance report")
    
    report = generate_performance_report(backtest_results)
    print(report)
    
    # Save report to file
    with open(f'{output_dir}/performance_report.txt', 'w') as f:
        f.write(report)
    
    # Save detailed results
    backtest_results['results_df'].to_csv(f'{output_dir}/backtest_results.csv', index=False)

    print(f" Results saved to: {output_dir}")

    
    return {
        'data': df,
        'regime_classifier': regime_classifier,
        'factor_predictor': factor_predictor,
        'strategy': strategy,
        'backtest_results': backtest_results,
        'regime_results': regime_results,
        'factor_results': factor_results
    }


if __name__ == "__main__":
    # Execute the pipeline
    results = main_pipeline(
        data_filepath='6_factor.xls',
        output_dir='outputs'
    )