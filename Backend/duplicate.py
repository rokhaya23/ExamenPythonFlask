# traitement des valeurs dupliquées
def duplicated_rows(df):
    initial_count=len(df)            # nombre d'enregistrement
    df_cleaned=df.drop_duplicates(keep='first')                #supression des doublons nouveau dataframe
    df_cleaned = df_cleaned.reset_index(drop=True)
    df_remove=initial_count - len(df_cleaned)              # nombre d'enregistrement supprimer

    stats={
        'initial_rows': initial_count,
        'final_rows': len(df_cleaned),
        'df_removed': df_remove,
        'pourcentage': round(df_remove/initial_count*100,2),
    }
    return df_cleaned, stats