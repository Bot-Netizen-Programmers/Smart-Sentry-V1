def label_phases(df, startup_cycles=10, shutdown_cycles=10):
    df['phase'] = 'steady_state'
    for engine_id in df['engine_id'].unique():
        engine_data = df[df['engine_id'] == engine_id]
        max_cycle = engine_data['cycle'].max()
        df.loc[(df['engine_id'] == engine_id) & (df['cycle'] <= startup_cycles), 'phase'] = 'startup'
        df.loc[(df['engine_id'] == engine_id) & (df['cycle'] > max_cycle - shutdown_cycles), 'phase'] = 'shutdown'
    return df

train_df = label_phases(train_df)
print(train_df[['engine_id', 'cycle', 'phase']].head(20))
