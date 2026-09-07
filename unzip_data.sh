
# Step 1: Create the target directory so tar has somewhere to extract into
mkdir -p timeseries_with_chronos/data/download

# Step 2: Read archive from $PWD, extract (-xf) into destination (-C)
tar -xf vitals_old.tar -C timeseries_with_chronos/data/download

# deleting the tar file
rm -f vitals_old.tar