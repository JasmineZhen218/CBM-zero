import os
def get_save_path(args):
    os.makedirs("saved_bb_features", exist_ok=True)
    os.makedirs("saved_clip_features", exist_ok=True)
    os.makedirs("saved_projections", exist_ok=True)
    os.makedirs("saved_clip_similarities", exist_ok=True)
    projection_path = "saved_projections/Data[{}]_ClassiModel[{}]_ClipModel[{}]_ConceptSource[{}]_Lambda[{}]_Power[{}].pt".format(
        args.data_name,
        args.black_box_model_name,
        args.clip_model_name,
        args.concept_set_source,
        args.lambd,
        args.power,
    )
    bb_features_train_path = "saved_bb_features/Data[{}]_Model[{}]_train.pt".format(
        args.data_name,
        args.black_box_model_name,
    )
    bb_features_val_path = "saved_bb_features/Data[{}]_Model[{}]_val.pt".format(
        args.data_name,
        args.black_box_model_name,
    )
    clip_text_embeddings_path = "saved_clip_features/Data[{}]_ClipModel[{}]_ConceptSource[{}]_text.pt".format(
        args.data_name,
        args.clip_model_name,
        args.concept_set_source,
    )
    clip_image_embeddings_train_path = "saved_clip_features/Data[{}]_ClipModel[{}]_ConceptSource[{}]_image_train.pt".format(
        args.data_name,
        args.clip_model_name,
        args.concept_set_source,
    )
    clip_image_embeddings_val_path = "saved_clip_features/Data[{}]_ClipModel[{}]_ConceptSource[{}]_image_val.pt".format(
        args.data_name,
        args.clip_model_name,
        args.concept_set_source,
    )

    clip_similarities_train_path = "saved_clip_similarities/Data[{}]_ClipModel[{}]_ConceptSource[{}]_Power[{}]_train.pt".format(
        args.data_name,
        args.clip_model_name,
        args.concept_set_source,
        args.power,
    )
    clip_similarities_val_path = "saved_clip_similarities/Data[{}]_ClipModel[{}]_ConceptSource[{}]_Power[{}]_val.pt".format(
        args.data_name,
        args.clip_model_name,
        args.concept_set_source,
        args.power,
    )
    clip_similarities_train_mean_path = "saved_clip_similarities/Data[{}]_ClipModel[{}]_ConceptSource[{}]_Power[{}]_train_mean.pt".format(
        args.data_name,
        args.clip_model_name,
        args.concept_set_source,
        args.power,
    )
    clip_similarities_train_std_path = "saved_clip_similarities/Data[{}]_ClipModel[{}]_ConceptSource[{}]_Power[{}]_train_std.pt".format(
        args.data_name,
        args.clip_model_name,
        args.concept_set_source,
        args.power,
    )
    return (projection_path, 
            bb_features_train_path, bb_features_val_path, 
            clip_text_embeddings_path, clip_image_embeddings_train_path, clip_image_embeddings_val_path, 
            clip_similarities_train_path, clip_similarities_val_path, clip_similarities_train_mean_path, clip_similarities_train_std_path)