#!/bin/bash -l
#SBATCH -t 17:00:00
#SBATCH -p archivelong 
#SBATCH -N 1
#SBATCH --mail-type=END,FAIL
#SBATCH -o /project/s/sievers/albatros/uapishka/202308/baseband/snap8/backup_uapishka_aug2023_quarry_%j.txt

# NO FORWARD SLASHES IN THE PATHS 
trap "echo 'Job script not completed';exit 129" TERM INT
if [ $# -ne 2 ]; then
    echo "Usage: $0 <Source dir (scratch)> <Target dir (archive)>"
    exit 1
fi
echo "Backing up files in $1 to $2"
shopt -s nullglob
SRC="$1"
DEST="$2"
maxfiles=10 #number of files in one tarball
cd $SRC
DIRS=(*)

#create 5 digit dirs in archive
for dir in ${DIRS[@]}
do
	echo "Checking for $dir"
	hsi ls "$DEST/$dir" &> /dev/null
	status=$?
	# echo $status
	if [[ $status == 0 ]]
	then
		echo "dir $dir exists at destination"
	else
		hsi mkdir "$DEST/$dir" &> /dev/null
		echo "Created dir $dir at destination"
	fi
done

#loop over each 5 digit dir
for d in $SRC/* #be careful, another / and stamp will fail (below)
do
        stamp=${d##*/} #get 5 digit sub-dir stamp by removing longest substring ending with /
	cd $d
	files=$(ls | sort -n)
	files=($files)
	numfiles=${#files[@]}
	echo "num of files in $d is" $numfiles
	#echo ${files[@]}
    	#loop over blocks of size maxfiles in each 5-digit dir
	for ((i=0;i<${#files[@]};i+=maxfiles))
	do
		tstamp=`echo ${files[$i]} | grep -o "[0-9]\{10\}"`
		fpath=$DEST/$stamp/$tstamp.tar
		hsi ls "$fpath" &> /dev/null
		status=$?
		if [ $status == 0 ]; then  
		    echo "File $fpath already exists. Continuing..."
		    continue
		fi
		# echo "file tstamp is" $tstamp
		# echo ${files[@]:$i:$maxfiles} # get relevant files
		htar -Humask=0137 -cpf "$fpath" -Hcrc -Hverify=1 "${files[@]:$i:$maxfiles}"
		status=$?
		if [ ! $status == 0 ]; then
		    echo 'HTAR returned non-zero code. while trying to tar files: ' "${files[@]:$i:$maxfiles}"
		    /scinet/niagara/bin/exit2msg $status
		    exit $status
		else
		    echo "$fpath TRANSFER SUCCESSFUL"
		fi
	done
done
