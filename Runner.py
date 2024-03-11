from .DataFetcher import download_files_from_blob_storage
from .DataGen import process_and_store_data



def synthesise_data(folder_name,update):
		print("Started The Downloadin")
		download_files_from_blob_storage("DefaultEndpointsProtocol=https;AccountName=llmaodatastore;AccountKey=J+go6cxXfi7v4Exq2ggINtxr13JICvHBoPjuKmGN5R0Ly6palZsO1RYiSOjvigxwx3jc1rOVPCI8+AStU/dpYQ==;EndpointSuffix=core.windows.net"
                                 ,folder_name, 
                                 "/home/llmao/fastapi/DataGen/corpus", 
                                 "data-corpus")

		print("data extracted \n")

		process_and_store_data(folder_name)

		print("structured data generated trained \n")
		update(dataGen="inactive")

