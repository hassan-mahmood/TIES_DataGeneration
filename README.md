# TIES Dataset Generation


* TableGeneration: Contains functionality to generate tables
* TFGeneration: Contains functionality to generate tfrecord files. It uses TableGeneration module to generate tables.
* generate_data: Main script to start dataset generation
* unlv_distribution: a binary file that contains words distribution of UNLV dataset (types of words: numbers, alphabets and words containing special characters)

## How to run

Use the following command to generate tfrecords:

```$ python generate_data.py --filesize num_of_images_per_tfrecord --threads num_of_threads --outpath output_directory_to_store_tfrecords --imagespath path/to/UNLV_images --ocrpath path/to/OCR_groundtruth_UNLV --tablepath path/to/UNLV_tables_ground_truths --writetoimg 0_or_1```


where,

**filesize**: Number of images to store in one tfrecord

**num_of_threads**: Threads are used to process files in parallel. A single thread generates one single tfrecord file. So 10 threads will generate 10 tfrecord files in parallel.

**outpath**: Output directory to store generated tfrecords

**writetoimg**: If writetoimg=1, the generated images will be stored (other than tfrecords) else not.

**imagespath**: Directory containing UNLV dataset images

**ocrpath**: Directory containing ground truths of characters in UNLV dataset

**tablepath**: Directory containing ground truths of tables in UNLV dataset


## Table Generation:

All of the content used in generated tables is extracted from UNLV dataset. We also extracted the distribution of alphabetical words, numbers and special character words from it and used same distribution for our dataset. Based on the distribution of words, we generated tables of 4 categories(as mentioned in the paper).

A table is generated in multiple steps like a lego building block(with each step contributing to generation of table):
1. We def the data types of columns e.g. which column will contain alphabets, numbers or special character words
2. Cells are randomly selected for missing data
3. Rows and Column spans are added to the table
4. The table is categorized into two ways(both categories are equally likely to be chosen):
    -   Table with regular headers(Table with only first row containing headers.)
    -   Table with irregular headers(Table with headers in first row and first column. This category can have multiple row spans for headers of first column.)
5. Table borders are chosen randomly. We define border_categories with 4 possibilities(all four categories are equally likely to be chosen):
    -   All borders
    -   No borders
    -   Borders only under headings
    -   Only internal borders
6. An equivalent HTML code is generated for this table.
7. This HTML coded table is converted to image using **selenium**.
6. Finally, shear and rotation transformations are applied to the table image.


## TFGeneration

During table generation process, the words are assigned unique IDs. During html-to-image conversion, these words are localized with bounding boxes and transformed on image transformation accordingly.

Based on these words IDs, we compute 3 adjacency matrices:
* **Same_Row**: if two IDs (e.g. 12, 25) are sharing a row, the value corresponding to that location(matrix[12][25]) will be 1 in that matrix.
* **Same_Column**: Matrix for column sharing IDs
* **Same_Cell**: Matrix for cell sharing IDs

Instead of just storing the image in tfrecord files, we also store some metadata:
1. Image height and width
2. Number of words in table
3. Table category
4. Word IDs
5. Adjacency matrix for same cells, same rows and same columns


