import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def select_csv_file():
    """Scans for CSV files in the numeric data folder and prompts the user to select one."""
    # Define the path to the numeric data folder
    numeric_data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'numeric')
    
    # Check if the folder exists, if not, create it
    if not os.path.exists(numeric_data_path):
        os.makedirs(numeric_data_path)
        print(f"Created numeric data folder at: {numeric_data_path}")
        print("Please add your CSV files to this folder and run the script again.")
        return None
    
    # Get CSV files from the numeric folder
    csv_files = [f for f in os.listdir(numeric_data_path) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in the numeric data folder: {numeric_data_path}")
        return None

    print(f"CSV files found in: {numeric_data_path}")
    print("\nPlease select a CSV file for advanced visualization:")
    for i, filename in enumerate(csv_files):
        print(f"{i + 1}: {filename}")

    while True:
        try:
            choice = int(input(f"Enter the number (1-{len(csv_files)}): ")) - 1
            if 0 <= choice < len(csv_files):
                # Return full path to the selected CSV file
                return os.path.join(numeric_data_path, csv_files[choice])
            else:
                print(f"Invalid number. Please enter a number between 1 and {len(csv_files)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def create_violin_plots(df, filename, output_dir):
    """Create violin plots for numeric variables."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        print("No numeric columns found for violin plots.")
        return
    
    n_cols = min(4, len(numeric_cols))
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    fig.suptitle(f'Violin Plots - Distribution Analysis - {filename}', fontsize=16, y=0.98)
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, col in enumerate(numeric_cols):
        row = i // n_cols
        col_idx = i % n_cols
        
        # Create violin plot
        parts = axes[row, col_idx].violinplot([df[col].dropna()], positions=[0], widths=0.8)
        
        # Customize violin plot colors
        for pc in parts['bodies']:
            pc.set_facecolor('lightblue')
            pc.set_alpha(0.7)
        
        axes[row, col_idx].set_title(f'{col}')
        axes[row, col_idx].set_ylabel('Values')
        axes[row, col_idx].grid(True, alpha=0.3)
        axes[row, col_idx].set_xticks([0])
        axes[row, col_idx].set_xticklabels([col])
    
    # Hide empty subplots
    for i in range(len(numeric_cols), n_rows * n_cols):
        row = i // n_cols
        col_idx = i % n_cols
        axes[row, col_idx].set_visible(False)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_violin_plots.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Violin plots saved to: {save_path}")

def create_kde_plots(df, filename, output_dir):
    """Create KDE plots for numeric variables."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        print("No numeric columns found for KDE plots.")
        return
    
    n_cols = min(4, len(numeric_cols))
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    fig.suptitle(f'KDE Plots - Density Estimation - {filename}', fontsize=16, y=0.98)
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, col in enumerate(numeric_cols):
        row = i // n_cols
        col_idx = i % n_cols
        
        # Create KDE plot
        df[col].dropna().plot.kde(ax=axes[row, col_idx], color='darkblue', linewidth=2)
        axes[row, col_idx].fill_between(axes[row, col_idx].get_lines()[0].get_xdata(), 
                                       axes[row, col_idx].get_lines()[0].get_ydata(), 
                                       alpha=0.3, color='lightblue')
        
        axes[row, col_idx].set_title(f'{col} - Kernel Density Estimation')
        axes[row, col_idx].set_xlabel(col)
        axes[row, col_idx].set_ylabel('Density')
        axes[row, col_idx].grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(len(numeric_cols), n_rows * n_cols):
        row = i // n_cols
        col_idx = i % n_cols
        axes[row, col_idx].set_visible(False)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_kde_plots.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"KDE plots saved to: {save_path}")

def perform_pca_analysis(df, filename, output_dir):
    """Perform PCA analysis and visualization."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) < 2:
        print("Need at least 2 numeric columns for PCA analysis.")
        return
    
    print("\n" + "="*80)
    print("PRINCIPAL COMPONENT ANALYSIS (PCA)")
    print("="*80)
    
    # Prepare data
    data_for_pca = df[numeric_cols].dropna()
    
    if len(data_for_pca) == 0:
        print("No complete cases available for PCA.")
        return
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_for_pca)
    
    # Perform PCA
    pca = PCA()
    pca_result = pca.fit_transform(scaled_data)
    
    # Calculate explained variance
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    print(f"\nExplained Variance by Principal Components:")
    print("-" * 50)
    for i, (var_ratio, cum_var) in enumerate(zip(explained_variance_ratio, cumulative_variance)):
        print(f"PC{i+1}: {var_ratio:.4f} ({var_ratio*100:.2f}%) | Cumulative: {cum_var:.4f} ({cum_var*100:.2f}%)")
    
    # Create PCA visualizations
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'PCA Analysis - {filename}', fontsize=16, y=0.98)
    
    # Scree plot
    ax1.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, 'bo-')
    ax1.set_title('Scree Plot')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.grid(True, alpha=0.3)
    
    # Cumulative variance plot
    ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'ro-')
    ax2.axhline(y=0.8, color='g', linestyle='--', label='80% threshold')
    ax2.axhline(y=0.95, color='orange', linestyle='--', label='95% threshold')
    ax2.set_title('Cumulative Explained Variance')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # PC1 vs PC2 scatter plot
    ax3.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.6, color='purple')
    ax3.set_title(f'PC1 vs PC2\n(Explains {(explained_variance_ratio[0] + explained_variance_ratio[1])*100:.1f}% of variance)')
    ax3.set_xlabel(f'PC1 ({explained_variance_ratio[0]*100:.1f}%)')
    ax3.set_ylabel(f'PC2 ({explained_variance_ratio[1]*100:.1f}%)')
    ax3.grid(True, alpha=0.3)
    
    # Feature importance heatmap for first 2 PCs
    components_df = pd.DataFrame(
        pca.components_[:2].T,
        columns=['PC1', 'PC2'],
        index=numeric_cols
    )
    
    sns.heatmap(components_df, annot=True, cmap='coolwarm', center=0, ax=ax4, cbar_kws={'shrink': 0.8})
    ax4.set_title('Feature Loadings (PC1 & PC2)')
    
    plt.tight_layout()
    plt.savefig(f'{os.path.splitext(filename)[0]}_pca_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return pca, pca_result

def create_categorical_analysis(df, filename, output_dir):
    """Create visualizations for categorical variables if present."""
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    if len(categorical_cols) == 0:
        print("No categorical columns found.")
        return
    
    print("\n" + "="*80)
    print("CATEGORICAL VARIABLES ANALYSIS")
    print("="*80)
    
    n_cols = min(3, len(categorical_cols))
    n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    fig.suptitle(f'Categorical Variables Analysis - {filename}', fontsize=16, y=0.98)
    
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, col in enumerate(categorical_cols):
        row = i // n_cols
        col_idx = i % n_cols
        
        if n_rows == 1:
            ax = axes[col_idx] if n_cols > 1 else axes[0]
        else:
            ax = axes[row, col_idx]
        
        # Count plot
        value_counts = df[col].value_counts().head(10)  # Top 10 categories
        value_counts.plot(kind='bar', ax=ax, color='lightcoral')
        ax.set_title(f'{col}\n(Top 10 categories)')
        ax.set_xlabel(col)
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Print statistics
        print(f"\n{col}:")
        print(f"  Unique values: {df[col].nunique()}")
        print(f"  Most common: {df[col].mode().iloc[0] if not df[col].mode().empty else 'N/A'}")
        print(f"  Missing values: {df[col].isnull().sum()}")
    
    # Hide empty subplots
    for i in range(len(categorical_cols), n_rows * n_cols):
        row = i // n_cols
        col_idx = i % n_cols
        if n_rows == 1:
            if n_cols > 1:
                axes[col_idx].set_visible(False)
        else:
            axes[row, col_idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{os.path.splitext(filename)[0]}_categorical_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_dashboard(df, filename, output_dir):
    """Create a summary dashboard with key insights."""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    fig.suptitle(f'EDA Summary Dashboard - {filename}', fontsize=20, y=0.98)
    
    # Dataset overview (top-left)
    ax1 = fig.add_subplot(gs[0, :2])
    overview_text = f"""
    Dataset Overview:
    • Shape: {df.shape[0]:,} rows × {df.shape[1]} columns
    • Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
    • Missing Values: {df.isnull().sum().sum():,} ({df.isnull().sum().sum()/df.size*100:.1f}%)
    • Numeric Columns: {len(df.select_dtypes(include=[np.number]).columns)}
    • Categorical Columns: {len(df.select_dtypes(include=['object']).columns)}
    • Duplicate Rows: {df.duplicated().sum():,}
    """
    ax1.text(0.1, 0.5, overview_text, transform=ax1.transAxes, fontsize=12, 
             verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    ax1.set_title('Dataset Overview', fontsize=14, fontweight='bold')
    
    # Missing values heatmap (top-right)
    ax2 = fig.add_subplot(gs[0, 2:])
    if df.isnull().sum().sum() > 0:
        sns.heatmap(df.isnull(), cbar=True, yticklabels=False, ax=ax2, cmap='viridis')
        ax2.set_title('Missing Values Pattern', fontsize=14, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'No Missing Values', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=16, fontweight='bold')
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        ax2.set_title('Missing Values Pattern', fontsize=14, fontweight='bold')
    
    # Data types distribution (middle-left)
    ax3 = fig.add_subplot(gs[1, :2])
    dtype_counts = df.dtypes.astype(str).value_counts()
    colors = plt.cm.Set3(np.linspace(0, 1, len(dtype_counts)))
    wedges, texts, autotexts = ax3.pie(dtype_counts.values, labels=dtype_counts.index, 
                                       autopct='%1.1f%%', colors=colors)
    ax3.set_title('Data Types Distribution', fontsize=14, fontweight='bold')
    
    # Numeric variables summary (middle-right)
    ax4 = fig.add_subplot(gs[1, 2:])
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        numeric_summary = df[numeric_cols].describe().loc[['mean', 'std', 'min', 'max']].T
        sns.heatmap(numeric_summary, annot=True, fmt='.2f', cmap='coolwarm', ax=ax4, cbar_kws={'shrink': 0.8})
        ax4.set_title('Numeric Variables Summary', fontsize=14, fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'No Numeric Variables', ha='center', va='center', 
                transform=ax4.transAxes, fontsize=16, fontweight='bold')
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('Numeric Variables Summary', fontsize=14, fontweight='bold')
    
    # Key insights (bottom)
    ax5 = fig.add_subplot(gs[2, :])
    insights_text = generate_key_insights(df)
    ax5.text(0.05, 0.95, insights_text, transform=ax5.transAxes, fontsize=11, 
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))
    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)
    ax5.axis('off')
    ax5.set_title('Key Insights', fontsize=14, fontweight='bold')
    
    plt.savefig(f'{os.path.splitext(filename)[0]}_summary_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_key_insights(df):
    """Generate key insights from the data."""
    insights = []
    
    # Data quality insights
    missing_pct = (df.isnull().sum().sum() / df.size) * 100
    if missing_pct > 20:
        insights.append(f"⚠️  High missing data rate: {missing_pct:.1f}% of values are missing")
    elif missing_pct > 5:
        insights.append(f"⚠️  Moderate missing data: {missing_pct:.1f}% of values are missing")
    else:
        insights.append(f"✅ Good data quality: Only {missing_pct:.1f}% missing values")
    
    # Duplicate rows
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        insights.append(f"⚠️  Found {duplicates:,} duplicate rows ({duplicates/len(df)*100:.1f}%)")
    
    # Numeric variables insights
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        insights.append(f"📊 {len(numeric_cols)} numeric variables available for analysis")
        
        # Check for potential outliers
        outlier_cols = []
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
            if len(outliers) > len(df) * 0.05:  # More than 5% outliers
                outlier_cols.append(col)
        
        if outlier_cols:
            insights.append(f"⚠️  Potential outliers detected in: {', '.join(outlier_cols[:3])}{'...' if len(outlier_cols) > 3 else ''}")
    
    # Categorical variables insights
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        insights.append(f"🏷️  {len(categorical_cols)} categorical variables found")
        
        high_cardinality = [col for col in categorical_cols if df[col].nunique() > len(df) * 0.1]
        if high_cardinality:
            insights.append(f"📈 High cardinality categories: {', '.join(high_cardinality[:2])}{'...' if len(high_cardinality) > 2 else ''}")
    
    # Memory usage
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    if memory_mb > 100:
        insights.append(f"💾 Large dataset: {memory_mb:.1f} MB memory usage")
    
    return '\n'.join([f"{i+1}. {insight}" for i, insight in enumerate(insights)])

def main():
    """Main function to run advanced visualization EDA."""
    filepath = select_csv_file()
    if not filepath:
        return
    
    try:
        df = pd.read_csv(filepath)
        filename = os.path.basename(filepath)
        output_dir = os.path.dirname(__file__)  # Save outputs in the numeric_EDA folder
        
        print(f"\nLoaded dataset: {filename}")
        
        # Update the file paths for saving visualizations
        def save_visualization(base_name, plot_type):
            return os.path.join(output_dir, f"{os.path.splitext(base_name)[0]}_{plot_type}.png")
        
        # Perform advanced visualizations with updated save paths
        create_violin_plots(df, filename, output_dir)
        create_kde_plots(df, filename, output_dir)
        perform_pca_analysis(df, filename, output_dir)
        create_categorical_analysis(df, filename, output_dir)
        create_summary_dashboard(df, filename, output_dir)
        
        print(f"\nAdvanced visualization EDA completed for {filename}")
        print(f"All visualizations saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error loading or processing file: {e}")

if __name__ == "__main__":
    main()