import argparse
import logging
import os.path
import shutil
import sqlite3
import time
import zipfile
import os
from StringIO import StringIO
from itertools import chain

from IPython import embed
from lxml import etree

# Config logger
logging.basicConfig(filename='logs/book_handler.log', level=logging.DEBUG)
logger = logging.getLogger(__name__)

def extract_images(path):
    data = []

    with open(path) as fxml:
        # Parse XML
        tree = etree.parse(fxml)

        # Extract namespace
        nstag = tree.getroot().tag
        namespace = nstag[1:].split('}')[0] if '}' in nstag else '' 

        # Find all imggroup
        imggroups = tree.findall('.//{%s}imggroup' % namespace)

        for group in imggroups:
            img = group.find('.//{%s}img' % namespace).get('src')
            caption = group.find('.//{%s}caption' % namespace)
            if caption is not None:
                caption = ([caption.text] + list(chain(*([c.text, etree.tostring(c), c.tail] for c in caption.getchildren()))) + [caption.tail])
                caption = ''.join(filter(None, caption)).encode('utf-8')

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
        img_original TEXT NOT NULL,
        caption TEXT,
        valid INTEGER DEFAULT 0,
        validated INTEGER DEFAULT 0,
        type INTEGER NULL
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

def handle_import(args):
    # File check
    if not zipfile.is_zipfile(args['filename']):
        logger.critical('The file "{}" is not a .zip file'.format(args['filename']))
        return

    # Open DB connection
    conn, cur = connect(args['db_file'])

    # Unzip
    xml_file = None
    with zipfile.ZipFile(args['filename']) as zf:
        zf.extractall(args['temp_dir'])
        # Find .xml file
        for name in zf.namelist():
            if name[-3:] == 'xml':
                xml_file = name

    if xml_file is None:
        logger.critical('No XML file in {}'.format(args['filename']))

    # Extract lois ID
    loisID = extract_lois('{}/{}'.format(args['temp_dir'], xml_file))
    logger.info('Parsing file {} with loisID {}'.format(args['filename'], loisID))

    # Check if not in database
    cur.execute("SELECT * FROM images WHERE lois_id = ?", (loisID,))
    if cur.fetchone():
        logger.info('{} already in the database!'.format(loisID))
    else:
        data = extract_images('{}/{}/{}.xml'.format(args['temp_dir'], loisID, loisID))

        for (img, caption) in data:
            img = img.replace('\\', '/') # fix windows-style paths
            if caption:
                caption = caption.decode('utf-8') # fix encoding of string

            # Move file to archive folder
            file_name_original = os.path.basename(img)
            file_name = '{}-{}.{}'.format(file_name_original.split('.')[0], time.time(), file_name_original.split('.')[1])
            source_path = '{}/{}/{}'.format(args['temp_dir'], loisID, img)
            dest_path   = '{}/{}'.format(os.path.join(args['temp_dir'], '..', 'images'), file_name)
            shutil.copyfile(source_path, dest_path)

            # Insert into DB
            cur.execute("INSERT INTO images (lois_id, img, img_original, caption) VALUES (?, ?, ?, ?)", (loisID, file_name, file_name_original, caption))

    # Close DB connection
    disconnect(conn)

    # Clean-up
    logger.info('Cleaning up {}'.format(args['filename']))
    os.remove(args['filename'])

def handle_export(args):
    if not args['loisid']:
        logger.error('LoisID required during export!')
        return

    conn, cur = connect(args['db_file'])

    # Check if in database
    cur.execute("SELECT * FROM images WHERE lois_id = ? AND validated = 1", (args['loisid'],))
    rows = cur.fetchall()
    if not rows:
        logger.info('{} not available in the database!'.format(args['loisid']))

    # Open book
    with open(os.path.join(args['temp_dir'], args['loisid'], '{}.xml'.format(args['loisid']))) as fxml:
        xml = fxml.read()
        xml_prolog = xml.split('<!DOCTYPE')[0]
        # Parse XML
        tree = etree.parse(StringIO(xml))

        # Extract namespace
        nstag = tree.getroot().tag
        ns = nstag[1:].split('}')[0] if '}' in nstag else '' 

    # Insert/modify captions into the XML
    for row in rows:
        _, _, _, img_name_original, caption, _, _, _ = row
        imggroup_elem = tree.xpath('.//ns:imggroup[.//ns:img[@src="img/{}"]]'.format(img_name_original), namespaces={'ns': ns})[0]
        caption_elem = imggroup_elem.xpath('.//ns:caption', namespaces={'ns': ns})
        if caption_elem:
            # Caption element already exists; simply change it
            caption_elem[0].text = caption
        else:
            # Construct caption element
            caption_elem = etree.Element('caption')
            caption_elem.text = caption
            imggroup_elem.append(caption_elem)

    # Zip 'm up
    logger.info('Creating zip...')
    zf = zipfile.ZipFile(args['filename'], mode='w')
    try:
        # Save the XML
        logger.info('Writing to XML file...')
        xml_out = os.path.join(os.path.join(args['temp_dir'], 'output'), '{}.xml'.format(args['loisid']))
        with open(xml_out, 'w') as fout:
            fout.write(xml_prolog)
            tree.write(fout, encoding='utf8', pretty_print=True)
        zf.write(xml_out, os.path.join(args['loisid'], '{}.xml'.format(args['loisid'])))

        # Move images to output structure
        logger.info('Exporting images...')
        cur.execute("SELECT * FROM images WHERE lois_id = ?", (args['loisid'],))
        rows = cur.fetchall()
        for row in rows:
            _, _, img_name_db, img_name_original, _, _, _, _ = row
            source_path = os.path.join(args['temp_dir'], '..', 'images', img_name_db)
            zf.write(source_path, os.path.join(args['loisid'], 'img', img_name_original))
    finally:
        logger.info('Saving zip to {}'.format(args['filename']))
        zf.close()

def main(args):
    # Create DB table(s) if needed
    if not os.path.isfile(args['db_file']) or args['rebuild']:
        init_db(args['db_file'])

    # Perform import / export action
    if args['i']:
        logger.info('Importing book...')
        handle_import(args)
    elif args['e']:
        logger.info('Exporting book...')
        handle_export(args)
    else:
        logger.error('Invalid mode! Valid modes: [import/export]')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Exctract images from XML file.')
    parser.add_argument('filename', help='Import/Export file (*.zip)')
    parser.add_argument('-i', help='Import file (*.zip)', action='store_true')
    parser.add_argument('-e', help='Export file (*.zip)', action='store_true')
    parser.add_argument('--loisid', '-l', help='LoisID to export')
    parser.add_argument('--rebuild', '-r', help='Rebuild database file', action='store_true')
    parser.add_argument('--temp_dir', default='temp', help='Temporary directory')
    parser.add_argument('--db_file', default='img.db', help='Database file')
    args = parser.parse_args()

    main(vars(args))
