from autogluon.multimodal.data.infer_types import infer_column_types

def prepare_df(df_train, df_test, df_dev, class_label, data_path):
    column_type = infer_column_types(
        data=df_train.drop(columns=[class_label, 'Image Path']),
        valid_data=df_dev.drop(columns=[class_label, 'Image Path'])
    )
    text_cols = [_[0] for _ in column_type.items() if _[1] in ['text', 'categorical']]
    df_train['combined_text'] = df_train[text_cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    df_test['combined_text'] = df_test[text_cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    df_dev['combined_text'] = df_dev[text_cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

    # set image path as column
    df_train['IMAGE PATH'] = df_train['Image Path'].apply(lambda x: data_path +  x)
    df_test['IMAGE PATH'] = df_test['Image Path'].apply(lambda x: data_path  + x)
    df_dev['IMAGE PATH'] = df_dev['Image Path'].apply(lambda x: data_path + x)

    return df_train, df_test, df_dev


def prepare_muldic(df, data_path):
    texts = list(df['title'])
    codes = list(df['code'])
    labels = list(df['label'])
    images = list(df['issue_num'].apply(lambda x: data_path + str(x) + '.png'))
    return texts, images, codes, labels