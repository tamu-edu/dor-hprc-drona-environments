cluster=`/sw/local/bin/clustername`

message="This EXPERIMENTAL LAMMPS environment will analyze your LAMMPS input file, provide a list of available LAMMPS modules, and generate a fully formed batch job based on the selected module. We recommend selecting PVC acceleration if possible to avoid excessive queue waiting times NOTE: If you face  any issues and/or have any suggestions, please contact us at help@hprc.tamu.edu."

if [[ "$cluster" != "aces" ]]; then
   message="This LAMMPS environment is current;y only available on the TAMU ACES cluster"
fi   
   
echo "<div style='background:Gainsboro; padding: 10px; border-radius: 7px; margin-bottom: 10px;'><p style='font-size: 1.1em; margin-bottom: 0;'><center>${message}</center></p></div>"
