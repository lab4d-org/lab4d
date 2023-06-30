# bash preprocess/scripts/download.sh <seqname>
seqname=$1
rootdir=$PWD

datadir=database/raw/$seqname
rm -rf $datadir
mkdir -p $datadir && cd "$_"
# download the video
wget $(cat $rootdir/database/vid_data/$seqname.txt) -O tmp.zip
unzip tmp.zip
rm tmp.zip
cd ../../
