# LORD Reproduce
## 1. Software Environment
1. python: 3.7.10
2. torch: 1.8.1+cu111
3. other package:
    ```
    pip install -r requirements.txt
    pip install signatory==1.2.0.1.4.0 --no-cache-dir --force-reinstall
    ```

## 2. Unzip the datasets
```
tar -zxvf data.tar
```

## 3. Reproduce the score of LORD and baselines
You can experiment two datasets, BIDMC32HR and BIDMC32RR.
1. Reproduce baseline's result
    ```
    cd baselines
    python main.py --model {MODEL} --ds_name {DATA} --D {DEPTH} --P {SUB-PATH LENGTH} --gpu {GPU}
    ```
    MODEL: baseline model, {odernn, ncde, ancde, nrde, denrde}  
    DATA: dataset, {BIDMC32HR, BIDMC32RR}  
    DEPTH: depth(valid only in nrde and denrde), {2,3}   
    SUB-PATH LENGTH: sub-path length, {1,8,128,512}  
    GPU: gpu number to use
    
2. Reproduce LORD's result
    ```
    cd lord
    python main.py --ds_name {DATA} --D1 {D1} --D2 {D2} --P {SUB-PATH LENGTH} --gpu {GPU}
    ```
    DATA: dataset, {BIDMC32HR, BIDMC32RR}  
    D1, D2: lower-depth and higher depth(D1 < D2), {2,3}  
    SUB-PATH LENGTH: sub-path length, {8,128,512}  
    GPU: gpu number to use

3. check the Result.  
    If you run `python check_score.py` in LORD folder, you can check all your scores.
    
    Model size is printed in terminal when training 