#!/bin/bash

# Make dir to store BBQ data
cd data
mkdir bbq_data
cd bbq_data

# Download all .jsonl files from https://github.com/nyu-mll/BBQ/tree/main/data
wget --no-check-certificate --content-disposition https://raw.githubusercontent.com/nyu-mll/BBQ/main/data/Age.jsonl .
wget --no-check-certificate --content-disposition https://raw.githubusercontent.com/nyu-mll/BBQ/main/data/Disability_status.jsonl .
wget --no-check-certificate --content-disposition https://raw.githubusercontent.com/nyu-mll/BBQ/main/data/Gender_identity.jsonl .
wget --no-check-certificate --content-disposition https://raw.githubusercontent.com/nyu-mll/BBQ/main/data/Nationality.jsonl .
wget --no-check-certificate --content-disposition https://raw.githubusercontent.com/nyu-mll/BBQ/main/data/Physical_appearance.jsonl .
wget --no-check-certificate --content-disposition https://raw.githubusercontent.com/nyu-mll/BBQ/main/data/Race_ethnicity.jsonl .
wget --no-check-certificate --content-disposition https://raw.githubusercontent.com/nyu-mll/BBQ/main/data/Religion.jsonl .
wget --no-check-certificate --content-disposition https://raw.githubusercontent.com/nyu-mll/BBQ/main/data/SES.jsonl .
wget --no-check-certificate --content-disposition https://raw.githubusercontent.com/nyu-mll/BBQ/main/data/Sexual_orientation.jsonl .
wget --no-check-certificate --content-disposition https://raw.githubusercontent.com/nyu-mll/BBQ/main/data/Race_x_SES.jsonl .
wget --no-check-certificate --content-disposition https://raw.githubusercontent.com/nyu-mll/BBQ/main/data/Race_x_gender.jsonl .

# Download supplementary material
wget --no-check-certificate --content-disposition https://raw.githubusercontent.com/nyu-mll/BBQ/main/supplemental/additional_metadata.csv .
mv additional_metadata.csv bbq_metadata.csv