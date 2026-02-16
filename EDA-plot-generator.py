import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io, base64, json, warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("Processed_Achareh_Orders_Sampled_Tehran_900000.csv")

plots = {}

def save_plot(name, fig=None):
    if fig is None:
        fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    plots[name] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close('all')
    print(f"  ✓ {name}")

print("Starting EDA...\n")

# ── 1. Target distribution (State) ──
print("[1/15] State distribution")
fig, ax = plt.subplots(figsize=(8, 5))
vc = df['State'].value_counts()
vc.plot.bar(ax=ax, color=sns.color_palette("viridis", len(vc)))
ax.set_title('Order State Distribution', fontsize=14)
ax.set_ylabel('Count')
for i, v in enumerate(vc.values):
    ax.text(i, v + len(df)*0.005, f'{v:,}\n({v/len(df)*100:.1f}%)', ha='center', fontsize=8)
plt.tight_layout()
save_plot('01_state_distribution')

# ── 2. Top 20 services ──
print("[2/15] Top services")
fig, ax = plt.subplots(figsize=(10, 6))
df['Service_Slug'].value_counts().head(20).plot.barh(ax=ax, color=sns.color_palette("mako", 20))
ax.set_title('Top 20 Services', fontsize=14)
ax.set_xlabel('Count')
ax.invert_yaxis()
plt.tight_layout()
save_plot('02_top20_services')

# ── 3. Top 15 categories ──
print("[3/15] Category distribution")
fig, ax = plt.subplots(figsize=(10, 5))
df['Category_Slug'].value_counts().head(15).plot.barh(ax=ax, color=sns.color_palette("rocket", 15))
ax.set_title('Top 15 Categories', fontsize=14)
ax.invert_yaxis()
plt.tight_layout()
save_plot('03_top15_categories')

# ── 4. Payment method ──
print("[4/15] Payment method")
fig, ax = plt.subplots(figsize=(6, 5))
df['PaymentMethod'].value_counts().plot.pie(ax=ax, autopct='%1.1f%%', startangle=90)
ax.set_title('Payment Method', fontsize=14)
ax.set_ylabel('')
plt.tight_layout()
save_plot('04_payment_method')

# ── 5. Gender distribution ──
print("[5/15] Gender")
fig, ax = plt.subplots(figsize=(6, 5))
df['Customer_Gender'].value_counts().plot.pie(ax=ax, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Set2"))
ax.set_title('Customer Gender', fontsize=14)
ax.set_ylabel('')
plt.tight_layout()
save_plot('05_gender')

# ── 6. Device distribution ──
print("[6/15] Device")
fig, ax = plt.subplots(figsize=(6, 5))
df['Device'].value_counts().plot.bar(ax=ax, color=sns.color_palette("pastel"))
ax.set_title('Device Distribution', fontsize=14)
ax.set_ylabel('Count')
plt.tight_layout()
save_plot('06_device')

# ── 7. Numeric distributions ──
print("[7/15] Numeric distributions")
num_cols = ['Selected_Price', 'Duration', 'Customer_CharehPoints', 'Customer_PreviousOrdersCount',
            'arranged_expert_rate', 'arranged_expert_successful_jobs', 'Time_to_First_Contract',
            'Time_to_Service', 'Customer_Account_Age', 'Customer_Order_Rate']
fig, axes = plt.subplots(2, 5, figsize=(22, 8))
for i, col in enumerate(num_cols):
    ax = axes[i//5, i%5]
    data = df[col].dropna()
    q01, q99 = data.quantile(0.01), data.quantile(0.99)
    data_clipped = data[(data >= q01) & (data <= q99)]
    ax.hist(data_clipped, bins=50, color=sns.color_palette("viridis", 10)[i], edgecolor='white', linewidth=0.3)
    ax.set_title(col.replace('_', '\n'), fontsize=9)
    ax.tick_params(labelsize=7)
fig.suptitle('Numeric Feature Distributions (1st-99th percentile)', fontsize=14, y=1.02)
plt.tight_layout()
save_plot('07_numeric_distributions')

# ── 8. Correlation heatmap ──
print("[8/15] Correlation heatmap")
num_df = df.select_dtypes(include=[np.number]).drop(columns=['Unnamed: 0'], errors='ignore')
corr = num_df.corr()
fig, ax = plt.subplots(figsize=(20, 16))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0, ax=ax, fmt='.1f',
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
            xticklabels=True, yticklabels=True)
ax.set_title('Feature Correlation Heatmap', fontsize=14)
ax.tick_params(labelsize=6)
plt.tight_layout()
save_plot('08_correlation_heatmap')

# ── 9. Top correlated pairs ──
print("[9/15] Top correlated pairs")
corr_pairs = corr.unstack().reset_index()
corr_pairs.columns = ['f1', 'f2', 'corr']
corr_pairs = corr_pairs[corr_pairs['f1'] < corr_pairs['f2']]
corr_pairs['abs_corr'] = corr_pairs['corr'].abs()
top30 = corr_pairs.nlargest(30, 'abs_corr')
fig, ax = plt.subplots(figsize=(10, 8))
labels = [f"{r['f1'][:20]} vs\n{r['f2'][:20]}" for _, r in top30.iterrows()]
colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in top30['corr']]
ax.barh(range(len(top30)), top30['corr'].values, color=colors)
ax.set_yticks(range(len(top30)))
ax.set_yticklabels(labels, fontsize=6)
ax.set_title('Top 30 Correlated Feature Pairs', fontsize=14)
ax.set_xlabel('Correlation')
ax.invert_yaxis()
plt.tight_layout()
save_plot('09_top_correlations')

# ── 10. Orders over time (Year + Season) ──
print("[10/15] Orders over time")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
df['Creation_DateTime_Year'].value_counts().sort_index().plot.bar(ax=axes[0], color='steelblue')
axes[0].set_title('Orders by Year')
axes[0].set_ylabel('Count')
season_map = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
df['Creation_DateTime_Season'].map(season_map).value_counts().reindex(['Spring','Summer','Fall','Winter']).plot.bar(ax=axes[1], color=sns.color_palette("coolwarm", 4))
axes[1].set_title('Orders by Season')
plt.tight_layout()
save_plot('10_orders_over_time')

# ── 11. Hourly patterns ──
print("[11/15] Hourly patterns")
fig, ax = plt.subplots(figsize=(12, 5))
df['Creation_DateTime_Hour'].value_counts().sort_index().plot(ax=ax, marker='o', color='steelblue', linewidth=2)
ax.set_title('Order Creation by Hour of Day', fontsize=14)
ax.set_xlabel('Hour')
ax.set_ylabel('Count')
ax.set_xticks(range(24))
ax.grid(True, alpha=0.3)
plt.tight_layout()
save_plot('11_hourly_pattern')

# ── 12. Day of week pattern ──
print("[12/15] Day of week")
fig, ax = plt.subplots(figsize=(8, 5))
dow_map = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
dow = df['Creation_DateTime_Day_of_Week'].map(dow_map).value_counts().reindex(['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])
dow.plot.bar(ax=ax, color=sns.color_palette("viridis", 7))
ax.set_title('Orders by Day of Week', fontsize=14)
ax.set_ylabel('Count')
plt.tight_layout()
save_plot('12_day_of_week')

# ── 13. Allocation type & Counselling ──
print("[13/15] Allocation & Counselling")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
df['AllocationType'].value_counts().plot.bar(ax=axes[0], color=sns.color_palette("Set2"))
axes[0].set_title('Allocation Type')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=15)
df['Counselling_Needed'].value_counts().plot.bar(ax=axes[1], color=['#2ecc71','#e74c3c'])
axes[1].set_title('Counselling Needed')
plt.tight_layout()
save_plot('13_allocation_counselling')

# ── 14. Price by State (boxplot) ──
print("[14/15] Price by State")
fig, ax = plt.subplots(figsize=(10, 5))
price_clip = df[df['Selected_Price'].between(df['Selected_Price'].quantile(0.01), df['Selected_Price'].quantile(0.95))]
sns.boxplot(data=price_clip, x='State', y='Selected_Price', ax=ax, palette='viridis')
ax.set_title('Selected Price by Order State (5th-95th pct)', fontsize=14)
ax.set_ylabel('Price')
plt.tight_layout()
save_plot('14_price_by_state')

# ── 15. Missing values ──
print("[15/15] Missing values")
miss = df.isnull().sum()
miss = miss[miss > 0].sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 6))
miss.plot.barh(ax=ax, color='salmon')
for i, v in enumerate(miss.values):
    ax.text(v + 500, i, f'{v/len(df)*100:.1f}%', va='center', fontsize=8)
ax.set_title('Missing Values by Column', fontsize=14)
ax.set_xlabel('Count')
ax.invert_yaxis()
plt.tight_layout()
save_plot('15_missing_values')

# ── Save to JSON file ──
with open('eda_plots.json', 'w') as f:
    json.dump(plots, f)

print(f"\n{'='*50}")
print(f"Done! {len(plots)} plots saved to eda_plots.json")
print(f"File size: {len(json.dumps(plots)) / 1024 / 1024:.1f} MB")
print(f"{'='*50}")
print("\nNow run: cat eda_plots.json | head -c 100")
print("To verify the file was created.")
