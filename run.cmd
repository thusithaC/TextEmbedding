spark-submit -v --master local[*] --conf "spark.driver.memory=16g"  --conf "spark.driver.maxResultSize=16"   pyspark_create_embeddings.py 
