from .DataFetcher import download_files_from_blob_storage
from .DataGen import process_and_store_data



def synthesise_data(folder_name,update):
		print("Started The Downloadin")
		download_files_from_blob_storage("BLOB_STORAGE_LINK"
                                 ,folder_name, 
                                 "/home/llmao/fastapi/DataGen/corpus", 
                                 "data-corpus")

		print("data extracted \n")

		process_and_store_data(folder_name)

		print("structured data generated trained \n")
		update(dataGen="inactive")

