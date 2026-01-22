"""
Visualization module for AirBnB rating prediction project.
"""

# TODO: Add visualization functions

# === Correlation Matrix ===

# Calculates correlations between numeric features and review_scores_rating, generates heatmap split by cities
def correlation_heatmap(df,
                        target='review_scores_rating',
                        cities =['LA', 'NY'],
                        exclude=None):
    
    global_corr = df.corr(numeric_only=True)[target]

    city_corrs = {}
    for city in cities:
        city_corrs[f"{city}_Correlation"] = df[df['city'] == city].corr(numeric_only=True)[target]
    
    correlations = pd.DataFrame({
    'Global_Correlation': global_corr,
    **city_corrs
    }).sort_values(by='Global_Correlation', ascending=False)

    if exclude:
        correlations = correlations.drop(index=exclude)
    
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(
        correlations,
        annot=True,
        cmap='coolwarm',
        center=0,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={'label': 'Correlation Coefficient'}
    )
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    plt.title(f'Correlation Comparison: {target}', fontsize=16)
    plt.ylabel('Features', fontsize=12)
    plt.xlabel('Market Split', fontsize=12)
    plt.tight_layout()
    plt.show()


# === Scatter Graph ===

# plots side by side scatter plots showing review scores vs selected features, seperated by city
def features_vs_rating_scatter_graph(df, features=None):
  plt.figure(figsize=(12, 8))

  for i, col in enumerate(features, 1):
    plt.subplot(2, 2, i)

    for city, color in zip(["NY", "LA"], ["tab:green", "tab:purple"]):
      subset = df[df["city"] == city]
      plt.scatter(
          subset[col],
          subset["review_scores_rating"],
          alpha=0.5,
          s=12,
          label=city,
          color=color)
  
  plt.title(f"Rating vs {col}")
  plt.xlabel(col)
  plt.ylabel("review_scores_rating")
plt.ylim(0, 5)
plt.legend()
plt.tight_layout()
plt.show()


# === Bar Chart ===

# Compares average ratings for hosts with and without selected binary features, grouped into a bar chart
def binary_host_attribute_comparison(df, features, target_col='review_scores_rating'):
    labels = []
    means_true = []
    means_false = []

    for feature in features:
        temp_df = df[[target_col, feature]].dropna()

        group_true = temp_df[temp_df[feature].isin([True, 't', 1, '1.0'])][target_col]
        group_false = temp_df[temp_df[feature].isin([False, 'f', 0, '0.0'])][target_col]

        if not group_true.empty and not group_false.empty:
            labels.append(feature)
            means_true.append(group_true.mean())
            means_false.append(group_false.mean())

    if not labels:
        return 

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, means_true, width, label='True / Yes', color='salmon', edgecolor='black', alpha=0.8)
    plt.bar(x + width/2, means_false, width, label='False / No', color='lightblue', edgecolor='black', alpha=0.8)
    plt.ylabel(f'Average {target_col}')
    plt.title('Impact of Host Features on Average Ratings', fontsize=14)
    plt.xticks(x, labels)
    all_means = means_true + means_false
    plt.ylim(min(all_means) - 0.05, max(all_means) + 0.05) 
    
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


# === Histogram ===

# Plots individual histograms for every numeric column in the dataframe
def plot_all_numeric_distributions(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns

    for col in numeric_columns:
        data=df[col].dropna()
        plt.figure(figsize=(8,4))
        sns.histplot(data, bins=50)
        plt.title(f'Distribution of {col} - Post Processing')
        plt.xlabel(col)
        plt.show()

# Plots a histogram of log-transformed values for selected numeric columns
def plot_log_transformed_histogram(df, column):
    log_data = np.log1p(df[column].dropna())
    
    bins = np.linspace(0, log_data.max(), 100)
    
    plt.figure(figsize=(8, 4))
    sns.histplot(log_data, bins=bins, color='skyblue')
    
    plt.title(f'Distribution of log {column}')
    plt.xlabel(f'log-transformed {column}')
    plt.ylabel('Frequency')
    plt.show()
    plt.close()

# Plots a histogram of the last review year for hosts who are unresponsive or have 0% acceptance
def plot_unresponsive_host_last_review_year(
    df,
    acceptance_col="host_acceptance_rate",
    response_col="host_response_rate",
    last_review_col="last_review",
    figsize=(10, 5),
    title="Last Review Year for Unresponsive / Non-Accepting Hosts"
):

    unresponsive_hosts = (
        (df[acceptance_col] == 0) |
        (df[response_col].isna())
    )

    unresponsive_hosts_last_review = df[last_review_col].dt.year.loc[unresponsive_hosts]

    min_year = unresponsive_hosts_last_review.min()
    max_year = unresponsive_hosts_last_review.max()
    bins = range(min_year, max_year + 2)

    plt.figure(figsize=figsize)
    sns.histplot(unresponsive_hosts_last_review, bins=bins, edgecolor="black")
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel("Number of Hosts")
    plt.xticks(range(min_year, max_year + 1), rotation=45)
    plt.show()


# === Cross-tabulation Chart ===

# Plots a stacked bar chart showing the percentage distribution of a categorical variable across groups
def plot_crosstab(
    df,
    index_col,
    category_col,
    title,
    ylabel="%",
    kind="bar",
    stacked=True
):
    (
        pd.crosstab(
            df[index_col],
            df[category_col],
            normalize="index"
        )
        .mul(100)
        .plot(kind=kind, stacked=stacked)
    )

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(index_col)
    plt.tight_layout()
    plt.show()

# === Box-plots ===

# Creates a boxplot comparing a numeric value across cities
def rating_by_city_boxplot(
    df,
    value_col,
    group_col,
    title,
    ylabel=None,
    figsize=(6, 4),
    showfliers=True
):

    plt.figure(figsize=figsize)

    df.boxplot(
        column=value_col,
        by=group_col,
        showfliers=showfliers
    )

    plt.title(title)
    plt.ylabel(ylabel or value_col)
    plt.xlabel(group_col)

    plt.tight_layout()
    plt.show()

# Plots log-price boxplots grouped by room type and city
def boxplot_log_price_by_room_city(
    df,
    room_types=None,
    price_col="log_price",
    room_type_col="room_type",
    city_col="city",
    figsize=(12, 5),
    showfliers=False
):

    if room_types is None:
        room_types = [
            "Entire home/apt",
            "Hotel room",
            "Private room",
            "Shared room"
        ]

    d = df[df[room_type_col].isin(room_types)].copy()

    label_map = {
        "Entire home/apt": "Entire",
        "Hotel room": "Hotel",
        "Private room": "Private",
        "Shared room": "Shared"
    }

    d["label"] = (
        d[room_type_col].replace(label_map)
        + " – "
        + d[city_col]
    )

    plt.figure(figsize=figsize)
    d.boxplot(column=price_col, by="label", showfliers=showfliers)
    plt.title("Log price by room type and city")
    plt.suptitle("")
    plt.ylabel(f"log({price_col.replace('log_', '')})")
    plt.xticks(rotation=25, ha="right")
    plt.show()
