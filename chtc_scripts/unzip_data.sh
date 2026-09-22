mkdir -p timeseries_with_chronos/data/download

# Labeled job: extract vitals_old.tar only if it exists.
if [ -f "vitals_old.tar" ]; then
    tar -xf vitals_old.tar -C timeseries_with_chronos/data/download
    rm -f vitals_old.tar
fi


# Unlabeled job: extract any site tar files that exist.
for tarfile in vitals_site_*_unlabelled.tar; do
    if [ -f "$tarfile" ]; then
        tar -xf "$tarfile" -C timeseries_with_chronos/data/download
        rm -f "$tarfile"
    fi
done


# Move outcomes.parquet only if it exists.
if [ -f "outcomes.parquet" ]; then
    mv outcomes.parquet timeseries_with_chronos/data/download/outcomes.parquet
fi

# Move death-gap parquet only if it exists (classify unlabeled sampling).
if [ -f "death_gap_results_updated.parquet" ]; then
    mv death_gap_results_updated.parquet timeseries_with_chronos/data/download/death_gap_results_updated.parquet
fi