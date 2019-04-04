"""
This script downloads speeches from the web.

Based on https://github.com/tomasz-oponowicz/spoken_language_dataset/tree/a4235d
4710377607f20a9b967ce3447e0c4a11a4
"""

import pandas as pd
import os
import hashlib
import shutil
import requests
import sys

GROUP_ATTR = 'Group'
LANGUAGE_ATTR = 'Language'
SEX_ATTR = 'Sex'
EXTENSION_ATTR = 'Extension'
URL_ATTR = 'Url'
MIRROR_ATTR = 'Mirror'


def fetch(url, output_file):
    print("Downloading {0}".format(url))
    response = requests.get(url, stream=True)
    with open(output_file, 'wb') as file:
        shutil.copyfileobj(response.raw, file)
    del response


def download(data, group, download_directory, create_lang_dir=False,
             create_group_dir=False):
    output_files = []

    data = pd.read_csv(data).fillna(value=False)
    for index, row in data.iterrows():
        if row[GROUP_ATTR] == group:
            language = row[LANGUAGE_ATTR]
            sex = row[SEX_ATTR][0]  # first letter, i.e. `f` or `m`
            extension = row[EXTENSION_ATTR]
            url = row[MIRROR_ATTR] or row[URL_ATTR]  # Prioritize mirrors
            url_hash = hashlib.md5(url.encode()).hexdigest()

            filename = "{lang}_{sex}_{url_hash}.{extension}".format(
                lang=language, sex=sex, url_hash=url_hash,
                extension=extension)
            if create_group_dir:
                download_directory += os.sep + group
            if create_lang_dir:
                download_directory += os.sep + language

            if not os.path.isdir(download_directory):
                os.makedirs(download_directory, exist_ok=True)

            output_file = os.path.join(download_directory, filename)
            output_files.append(output_file)
            if output_file not in os.listdir(download_directory):
                fetch(url, output_file)


if __name__ == '__main__':
    import argparse

    # Command line arguments:
    parser = argparse.ArgumentParser(description='Speech files downloader')
    parser.add_argument('speeches', help='Speech CSV file')
    parser.add_argument('output', help='Output directory')
    parser.add_argument('group', help='train or test')
    parser.add_argument('--separate_by_lang', help='Separate downloaded files '
                                                   'by language',
                        action='store_true')
    parser.add_argument('--separate_by_group', help='Separate downloaded files '
                                                    'by language',
                        action='store_true')

    arguments = parser.parse_args()
    download(data=arguments.speeches, group=arguments.group,
             download_directory=arguments.output,
             create_lang_dir=arguments.separate_by_lang,
             create_group_dir=arguments.separate_by_group)
