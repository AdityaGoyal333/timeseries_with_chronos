mkdir -p timeseries_with_chronos/data/download

# Labeled job: mats. Unlabeled job: this file is not transferred.
tar -xf vitals_old.tar -C timeseries_with_chronos/data/download
rm -f vitals_old.tar


# Unlabeled job: only that site's tar is in cwd. Archive already contains site_i/.
for tarfile in vitals_site_*_unlabelled.tar; do
    tar -xf "$tarfile" -C timeseries_with_chronos/data/download
    rm -f "${tarfile}.tar"
done


# outcomes.parquet lands in job cwd; pipeline reads data/download/outcomes.parquet
mv outcomes.parquet timeseries_with_chronos/data/download/outcomes.parquet
