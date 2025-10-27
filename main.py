def main():
    # Import custom modules
    from src.loader import CorpusLoader
    from src.embedder import Embedder
    from src.projecter import ProjectionAnalyzer
    from src.pipeline import ConceptProjector
    from src.plotter import Plotter
    print("Modules imported successfully.")
    # Define project name:
    project_name = "Fiction4Sentiment"

    # --- Load Classes, Split Data and Investigate the Data ---
    # --- Initialize Embedder Class:
    MultiLingMPNET = Embedder(model_name="paraphrase-multilingual-mpnet-base-v2")

    # --- Initialize Loader Class:
    # Load the fiction4 dataset from HuggingFace -- Training set (positive/negative filter)
    Fiction4Loader = CorpusLoader(text_col="text", label_col="label")
    Fiction4Loader.load_from_huggingface("chcaa/fiction4sentiment", split="train")
    Fiction4Loader.split_binary_train_continuous_test(positive_threshold=7, negative_threshold=3, train_size=0.6, random_state=42) # Define the thresholds for positive and negative labels, and split the non-neutral elements into training and test sets.

    # Load the translated fiction4 dataset from csv -- Test set (continuous labels)
    Fiction4TranslatedLoader = CorpusLoader(path="data/raw/fiction4sentiment_saved_translations.csv", text_col="text", label_col="label")
    Fiction4TranslatedLoader.load_csv()
    Fiction4TranslatedLoader.only_projection_no_split()

    # Load the EmoBank Corpus
    EmobankLoader = CorpusLoader(path="data/raw/emobank.csv", text_col="text", label_col="V")
    EmobankLoader.load_csv()
    # Append emobank metadata
    import pandas as pd
    df_meta = pd.read_csv("data/raw/emobank_meta.tsv", sep="\t")
    EmobankLoader.df = pd.merge(EmobankLoader.df, df_meta, left_on="id", right_on="id", how="left")
    EmobankLoader.only_projection_no_split()
    print("Data loaded and split successfully.")

    # --- Initialize the Pipeline Class ---
    # --- Project the Fiction4 Dataset ---
    pipeline_Fiction4 = ConceptProjector(
        CorpusLoader=Fiction4Loader,
        Embedder=MultiLingMPNET,
        category_col="category", 
        project_name="Fiction4Sentiment")
    pipeline_Fiction4.run()

    # --- Project the translated Fiction4 Dataset ---
    pipeline_Fiction4_Translated = ConceptProjector(
        CorpusLoader=Fiction4TranslatedLoader,
          Embedder=MultiLingMPNET, 
          category_col="category",
          project_name="Fiction4Sentiment_Translated", 
          use_saved_concept_vector=True, 
          concept_vector_path = pipeline_Fiction4.concept_vector_path)
    pipeline_Fiction4_Translated.run()

    # --- Project the EmoBank Dataset ---
    pipeline_Emobank = ConceptProjector(
        CorpusLoader=EmobankLoader,
          Embedder=MultiLingMPNET, 
          category_col="category",
          project_name="EmobankSentiment", 
          use_saved_concept_vector=True, 
          concept_vector_path = pipeline_Fiction4.concept_vector_path)
    pipeline_Emobank.run()
    print("Pipelines executed successfully.")
    
       
    # --- Plot the Results ---
    plots = Plotter()

    # --- Plot the Results for Fiction4 ---
    plots.marginal_scatterplot(projections=pipeline_Fiction4.results["prediction"], labels=pipeline_Fiction4.results["label"],
                         xlabel='Fiction4 Subspace Projection (Standardized)', 
                         y_label='Human Gold Standard', 
                         title='Scatterplot with Correlation (MPNET Standardized)',
                         save_path='plots/Scatterplot_fiction4_MultiLingMPNET_standardized.pdf')
    
    plots.category_correlation_table(df=pipeline_Fiction4.results, 
                                     category_col="category", 
                                     label_col="label", 
                                     pred_col="prediction",
                                     dataset="Fiction4",
                                     save_path='plots/Category_Correlation_Table_Fiction4_MultiLingMPNET.pdf')
    plots.category_correlation_table(df=pipeline_Fiction4_Translated.results, 
                                     category_col="category", 
                                     label_col="label", 
                                     pred_col="prediction",
                                     dataset="Fiction4 - Translated (Da -> En)",
                                     save_path='plots/Category_Correlation_Table_Fiction4_Translated_MultiLingMPNET.pdf')
    
    # --- Add a language column based on category ---
    pipeline_Fiction4.results["language"] = pipeline_Fiction4.results["category"].apply(lambda x: "Danish" if x in ["poetry", "prose"] else "English")
    plots.category_correlation_table(df=pipeline_Fiction4.results, 
                                     category_col="language", 
                                     label_col="label", 
                                     pred_col="prediction",
                                     dataset= "Fiction4 by Language",
                                     save_path='plots/Language_Correlation_Table_Fiction4_MultiLingMPNET.pdf')
    
    # --- Plot the Results for Emobank ---
    plots.marginal_scatterplot(projections=pipeline_Emobank.results["prediction"], labels=pipeline_Emobank.results["label"],
                         xlabel='Emobank Subspace Projection (Standardized)', 
                         y_label='Human Gold Standard', 
                         title='Scatterplot with Correlation (MPNET Standardized)',
                         save_path='plots/Scatterplot_Emobank_MultiLingMPNET_standardized.pdf')
    
    plots.category_correlation_table(df=pipeline_Emobank.results, 
                                     category_col="category", 
                                     label_col="label", 
                                     pred_col="prediction",
                                     dataset="Emobank",
                                     save_path='plots/Category_Correlation_Table_Emobank_MultiLingMPNET.pdf')
    print("Plots generated and saved successfully.")

if __name__ == "__main__":
    main()


