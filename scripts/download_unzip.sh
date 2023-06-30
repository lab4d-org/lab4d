# bash scripts/download_unzip.sh "$url"
url=$1
rootdir=$PWD

# download the dataset
wget $url -O tmp.zip
unzip tmp.zip
rm tmp.zip
