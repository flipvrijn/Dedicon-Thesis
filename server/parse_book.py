import argparse
import logging
import os.path
import shutil
import sqlite3
import time
import zipfile

from IPython import embed
from lxml import etree

# Config logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def extract_images(path):
    data = []

    with open(path) as fxml:
        # Parse XML
        tree = etree.parse(fxml)

        # Extract namespace
        namespace = tree.getroot().tag[1:].split('}')[0]

        # Find all imggroup
        imggroups = tree.findall('.//{%s}imggroup' % namespace)

        for group in imggroups:
            img = group.find('.//{%s}img' % namespace).get('src')
            caption = group.find('.//{%s}caption' % namespace)
            if caption is not None:
                caption = ''.join([text for text in caption.itertext()]).encode('utf-8')

            data.append((img, caption))

    return data

def extract_lois(path):
    with open(path) as fxml:
        # Parse XML
        tree = etree.parse(fxml)

        # Extract namespace
        namespace = tree.getroot().tag[1:].split('}')[0]

        meta = tree.find('.//{%s}meta[@name="dtb:uid"]' % namespace)
        if meta is None:
            logger.critical('XML file has no lois ID!')

        return meta.values()[1]

def init_db(db_file):
    logger.info('Initializing database...')
    conn = sqlite3.connect(db_file)
    c    = conn.cursor()
    c.execute("DROP TABLE IF EXISTS images")
    c.execute('''CREATE TABLE images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        lois_id TEXT NOT NULL,
        img TEXT NOT NULL,
        caption TEXT,
        valid INTEGER DEFAULT 0,
        validated INTEGER DEFAULT 0
    )''')
    conn.commit()
    conn.close()
    logger.info('Database created!')

def connect(db_file):
    logger.info('Connecting to {}'.format(db_file))
    conn = sqlite3.connect(db_file)
    c    = conn.cursor()

    return (conn, c)

def disconnect(connection):
    connection.commit()
    connection.close()
    logger.info('Disconnected!')

def main(args):
    # File check
    if not zipfile.is_zipfile(args.in_file):
        logger.critical('The input file "{}" is not a .zip file'.format(args.in_file))
        return

    # Create DB table(s) if needed
    if not os.path.isfile(args.db_file) or args.rebuild:
        init_db(args.db_file)

    # Open DB connection
    conn, cur = connect(args.db_file)

    # Unzip
    xml_file = None
    with zipfile.ZipFile(args.in_file) as zf:
        zf.extractall(args.temp_dir)
        # Find .xml file
        for name in zf.namelist():
            if name[-3:] == 'xml':
                xml_file = name

    if xml_file is None:
        logger.critical('No XML file in {}'.format(args.in_file))

    # Extract lois ID
    loisID = extract_lois('{}/{}'.format(args.temp_dir, xml_file))
    logger.info('Parsing file with loisID {}'.format(loisID))

    # Check if in database
    cur.execute("SELECT * FROM images WHERE lois_id = ?", (loisID,))
    if cur.fetchone():
        logger.info('{} already in the database!'.format(loisID))
    else:
        data = extract_images('{}/{}/{}.xml'.format(args.temp_dir, loisID, loisID))

        for (img, caption) in data:
            img = img.replace('\\', '/') # fix windows-style paths
            if caption:
                caption = caption.decode('utf-8') # fix encoding of string

            # Move file to archive folder
            file_name = os.path.basename(img)
            file_name = '{}-{}.{}'.format(file_name.split('.')[0], time.time() ,file_name.split('.')[1])
            source_path = '{}/{}/{}'.format(args.temp_dir, loisID, img)
            dest_path   = '{}/{}'.format(os.path.join(args.temp_dir, '..', 'images'), file_name)
            shutil.copyfile(source_path, dest_path)

            # Insert into DB
            cur.execute("INSERT INTO images (lois_id, img, caption) VALUES (?, ?, ?)", (loisID, file_name, caption))

    # Close DB connection
    disconnect(conn)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Exctract images from XML file.')
    parser.add_argument('in_file', help='Input file (*.zip)')
    parser.add_argument('--rebuild', '-r', help='Rebuild database file', action='store_true')
    parser.add_argument('--temp_dir', default='temp', help='Temporary directory')
    parser.add_argument('--db_file', default='img.db', help='Database file')
    args = parser.parse_args()

    main(args)