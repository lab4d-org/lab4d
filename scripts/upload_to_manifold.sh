filename=dog-2024-02-19
python scripts/zip_dataset.py $filename;
manifold -vip -cert /mnt/home/$USER/my-user-cert.pem put $filename.zip codec-avatars-scratch/tree/gengshany/$filename.zip

filename=bunny-2024-02-14
python scripts/zip_dataset.py $filename;
manifold -vip -cert /mnt/home/$USER/my-user-cert.pem put $filename.zip codec-avatars-scratch/tree/gengshany/$filename.zip

filename=human-2024-05-15
python scripts/zip_dataset.py $filename;
manifold -vip -cert /mnt/home/$USER/my-user-cert.pem put $filename.zip codec-avatars-scratch/tree/gengshany/$filename.zip
