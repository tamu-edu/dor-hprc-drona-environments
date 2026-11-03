cluster=`/sw/local/bin/clustername`

message="This LAMMPS environment analyzes your LAMMPS input file, suggests available LAMMPS modules, and generates a complete batch job based on the selected module. We recommend selecting PVC acceleration, if possible, to avoid excessive queue wait times. NOTE: Help us improve the LAMMPS environment. If you have any issues or have suggestions, please fill out a form by clicking the feedback link in the footer. You can also use the form if you want a specialized version of the LAMMPS environment or if no software builds are available for your input."

if [[ "$cluster" != "aces" ]]; then
   message="This LAMMPS environment is currently only available on the TAMU ACES cluster"
fi   
   
echo "<div style='background:Gainsboro; padding: 10px; border-radius: 7px; margin-bottom: 10px;'><p style='font-size: 1.1em; margin-bottom: 0;'><center>${message}</center></p></div>"
