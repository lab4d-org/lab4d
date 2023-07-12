# bash scripts/download_unzip.sh "$url"
url=$1
rootdir=$PWD

filename=tmp-`date +"%Y-%m-%d-%H-%M-%S"`.zip
wget $url -O $filename
unzip $filename
rm $filename
