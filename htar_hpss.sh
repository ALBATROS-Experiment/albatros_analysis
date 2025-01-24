#!/bin/bash -l
#BACKUP SPEED IS ROUGHLY 3.33 HR/TB
#SBATCH -t 25:00:00
#SBATCH -p archivelong 
#SBATCH -N 1
#SBATCH --mail-type=END,FAIL
#SBATCH -o /project/s/sievers/albatros/mars/mars202207_sep_2022_%j.txt

trap "echo 'Job script not completed';exit 129" TERM INT
if [ $# -ne 2 ]; then
    echo "Usage: $0 <Source dir (scratch)> <Target dir (archive)>"
    exit 1
fi
echo "Backing up files in $1 to $2"
shopt -s nullglob
SRC=$(realpath "$1")
DEST=${2%/} #remove trailing slashes and relative paths
maxfiles=10 #number of files in one tarball
DIRS=( $(find "$SRC" -mindepth 1 -type d -regextype posix-extended -regex ".*/[0-9]{5}"| sort -n) ) #find valid 5-digit dirs : absolute paths

#create 5 digit dirs in archive
for dir in ${DIRS[@]}
do
    stamp=${dir##*/} #get 5 digit sub-dir stamp by removing longest substring ending with /
    echo "Checking 5-digit tstamp $stamp of $dir"
	hsi ls "$DEST/$stamp" &> /dev/null
    #ls "$DEST/$stamp" &> /dev/null
	status=$?
	# echo $status
	if [[ $status == 0 ]]
	then
		echo "dir $stamp exists at destination"
	else
		hsi mkdir "$DEST/$stamp" &> /dev/null
        #mkdir "$DEST/$stamp" &> /dev/null
                status=$?
                echo "status is $status"
                if [[ $status == 0 ]]
                then
		    echo "Created dir $stamp at destination"
                else
                    echo "Dir creation failed on tape"
                    exit $status
                fi
	fi
done

#loop over each 5 digit dir
for d in ${DIRS[@]} #be careful, another / and stamp will fail (below)
do
        d=${d%/} #remove trailing slash if something went wrong earlier
        stamp=${d##*/} #get 5 digit sub-dir stamp by removing longest substring ending with /
	echo "cd into $d"
        cd $d
	files=$(ls | sort -n)
	files=($files)
	numfiles=${#files[@]}
	echo "num of files in $d is" $numfiles
	# echo ${files[@]}
    	#loop over blocks of size maxfiles in each 5-digit dir
	for ((i=0;i<${#files[@]};i+=maxfiles))
	do
		tstamp=`echo ${files[$i]} | grep -o "[0-9]\{10\}"`
		fpath=$DEST/$stamp/$tstamp.tar
        #echo $fpath
	        hsi ls "$fpath" &> /dev/null
        #ls "$fpath" &> /dev/null
		status=$?
		if [ $status == 0 ]; then  
		    echo "File $fpath already exists. Continuing..."
		    continue
		fi
		# echo "file tstamp is" $tstamp
		# echo ${files[@]:$i:$maxfiles} # get relevant files
		htar -Humask=0137 -cpf "$fpath" -Hcrc -Hverify=1 "${files[@]:$i:$maxfiles}"
                #tar -cpf "$fpath" "${files[@]:$i:$maxfiles}"
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
