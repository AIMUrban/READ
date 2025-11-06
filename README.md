# Urban Region Embedding with Adaptive Region Correlation Discovery

These are the source codes for the **READ** model and its corresponding data.

- Data
  
  The files in the data/ folder are the data used to train the **READ** model and perform downstream tasks.
  
- Code
  
  1. READ.py is the code of the main structure of the **READ** model.
  2. layers.py is the code including the implementation details of all components of **READ**.
  3. train.py is executed to train the model.
  4. tasks.py is the code for three downstream tasks, containing region crime prediction, region popularity prediction, and land usage clustering.
  5. validation.py can get the results on different downstream tasks.
  
  
  
- Urban Region Embedding

  save_emb/ — These are the learned embeddings of urban regions. Their data form is a 270x128 matrix,  where 270 is the number of regions and 128 is the dimension size of the region embedding.

