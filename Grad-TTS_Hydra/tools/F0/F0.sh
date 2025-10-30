job_name=$1
itv=1 #Epoch interval of evaluationg the F0
root_directory="/exp/exp4/acp23xt/Hydra_${job_name}_eval/$2$3/dev" #The directory that contains the folders for each wavs file from evaluation step
#root_directory="/exp/exp4/acp23xt/Hydra_${job_name}_eval/$2$3/dev"

Gt_directory="${root_directory}/ground_truth"
Converted_directory="${root_directory}/converted" #it containes N epoch subdirectories, each comparing with Ground truth directory
Output_directory="${root_directory}/metrics"
[ -d ${Output_directory} ] || mkdir -p "${Output_directory}"

#When dealing with F0 of a folder containing subfolders

sorted_folder=$(python "tools/F0/sorted.py" -d $Converted_directory) #sort the epoch directory according to epoch numbers

folder_list=($sorted_folder)

length=${#folder_list[@]}

rm  "$Output_directory/mean_f0.txt"
rm  "$Output_directory/std_f0.txt"
rm  "$Output_directory/F0.txt"

# Loop through the folder_list with an interval of 10
for ((i=0; i<length; i+=itv)); do
    folder=${folder_list[$i]}
    echo "$folder"
    python "tools/F0/F0.py" --gt_wavdir_or_wavscp "$Gt_directory" --gen_wavdir_or_wavscp  "$folder" --outdir "$Output_directory"

done

python "tools/F0/line_chart.py" -p $Output_directory/mean_f0.txt -s 0 -i $itv -t F0 -x Epoch -y mean_F0 -o $Output_directory/mean_F0.png
python "tools/F0/line_chart.py" -p $Output_directory/std_f0.txt -s 0 -i $itv -t F0 -x Epoch -y std_F0 -o $Output_directory/std_F0.png
