# ASMNeoPolyp20241

Best_model_link:https://drive.google.com/file/d/17_x_oimwTMf5nYlxmNfO3FyOJXV6D4kz/view?usp=sharing

The folder ***data*** contains initial files, which are ***train***, ***train_gh***, ***test*** from [BKAI-IGH NeoPolyp/Data](https://www.kaggle.com/competitions/bkai-igh-neopolyp/data)  . You should download the folder on this link and *Extract* all the files.

In the Working Directory, you should put a folder name ***data*** if you want to *modify* and run the ***.ipynb*** type code. Inside ***data*** are three folders ***train***, ***train_gh***, ***test***. Each folder contain images corresponding to the data of the challenge.

[Best-model-checkpoint](https://drive.google.com/file/d/17_x_oimwTMf5nYlxmNfO3FyOJXV6D4kz/view?usp=sharing) is store here, you must download it and place it inside folder ***code***.

Then, you can run the file ***infer.py*** smoothly.

Here is the steps:
Open your git bash or terminal and do these successively:
-	git clone https://github.com/bluff-king/ASMNeoPolyp20241.git
-	cd ASMNeoPolyp20241
-	download the [Best-model-checkpoint](https://drive.google.com/file/d/17_x_oimwTMf5nYlxmNfO3FyOJXV6D4kz/view?usp=sharing) and place it inside code
-	python3 infer.py --image_path image.jpeg
