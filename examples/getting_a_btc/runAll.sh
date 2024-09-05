foamCloneCase ../../templates/first_order_adsorption/pfasFoam my_pfas_case
pfasFoam -case my_pfas_case
touch my_pfas_case/case.foam
python make_report.py my_pfas_case