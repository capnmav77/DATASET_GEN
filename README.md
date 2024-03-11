# DATASET_GEN: Unstructured to Structured Data Conversion for HuggingFace

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)

## Introduction

Welcome to **DATASET_GEN**, a powerful tool designed to transform unstructured data into a structured format suitable for machine learning and natural language processing tasks. This project is particularly useful for those working with large datasets that require preprocessing before being utilized in models, such as those available on HuggingFace.

## Features

- **Data Chunking**: Efficiently splits large unstructured datasets into manageable chunks.
- **Structured Data Conversion**: Converts unstructured data into a structured format, such as JSON or CSV, for easier processing.
- **HuggingFace Compatibility**: Directly uploads structured data to HuggingFace, enabling seamless integration with HuggingFace models.
- **Customizable**: Offers flexibility in data processing and conversion, allowing for tailored solutions to specific dataset needs.

## Installation

To get started with **DATASET_GEN**, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/capnmav77/DATASET_GEN.git
   ```
2. Navigate to the project directory:
   ```
   cd DATASET_GEN
   ```
3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

**DATASET_GEN** is designed to be user-friendly. Here's a basic workflow:

1. **Prepare Your Data**: Ensure your unstructured data is ready and accessible.
2. **Configure the Script**: Modify the `Data_fetcher.py` file to specify your data source, chunk size, and structured data format.
3. **Configure the Script**: Modify the `DataGen.py` file to specify your chunk size, structured data format and the hugginface upload path.
4. **Run the Conversion**: Execute the main script to process your data:
   ```
   python Runner.py
   ```
5. **Upload to HuggingFace**: Once your data is structured, use the provided utility to upload it directly to HuggingFace.

## Contributing

We welcome contributions from the community! If you're interested in contributing to **DATASET_GEN**, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them to your branch.
4. Open a pull request with a detailed description of your changes.

---

**DATASET_GEN** is a project that aims to simplify the process of converting unstructured data into a structured format, making it easier to work with machine learning models, especially those available on HuggingFace. Whether you're working on a project that requires large-scale data processing or simply looking to streamline your workflow, **DATASET_GEN** has you covered.

Citations:
[1] https://github.com/adithya-s-k/LLM-Alchemy-Chamber
[2] https://huggingface.co/LLMao

