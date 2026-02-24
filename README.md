

# How Safe Will I Be Given What I Saw? 


**Calibrated Prediction of Safety Chances for Image-Controlled Autonomy**

This is the code for paper: "How Safe Am I Given What I See?  Calibrated Prediction of Safety Chances for Image-Controlled Autonomy". This repository contains the following: 
* The DQN controller training code
* Trained controllers for the racing car and the cart pole
* Data generation code
* Training code for all of our evaluators, autoencoders, forecasters, and predictors
* Conformal calibration code 


## Prerequisites

```python
pip install -r requirements.txt
```

The version of gym is 0.21.0 and box2d-py is 2.3.8.

## Train the controllers

The code to train the racing car controller is in the directory RacingCarController(https://github.com/andywu0913/OpenAI-GYM-CarRacing-DQN) 

The code to train the cart pole in the directory CartPoleController(https://github.com/fedebotu/vision-cartpole-dqn)

## Data generation

Before collecting data, we need to edit the car_racing.py files in PYTHON_PATH/site-packages/gym/envs/box2d/car_racing.py to get the position of the car at each moment.

In def step(self,cation), just change the return value to:

```python
return self.state, step_reward, done, [self.car.hull.position,self.road_poly,self.observation_space]
```
or directly substitute the file with EditedGym/car_racing.py

In the experiments on the racing car, we used 6 different models and collected 80K training samples, 20K test samples, 20K calibration samples, and 20K validation samples for each controller. 

The example is here:

```bash
python RacingCar/gene_data.py --n=80000 -d="RacingCar/data/train/controller_6/" -c="RacingCar/models/trial_600.h5" -s=0
```
where n is the number of samples, d is the dir of output path, c is the path of DQN models, and s is the random seed.

For the cart pole experiments, we used 3 different models and collect 30K training samples, 30K test samples, 30K calibration samples, and 30K validation samples for each controller

```python
    return np.array(self.state, dtype=np.float32), reward, done, theta
```

```bash
python CartPole/gene_data.py --n=30000 -d="CartPole/data/train/controller_1/" -c="CartPole/models/policy_net_best1.pt" -s=0
```

## Evaluator training

```bash
python RacingCar/evaluator/train.py --log="RacingCar/evaluator/logs/" --train="RacingCar/data/train/" --test="RacingCar/data/test/"
```


```bash
python CartPole/evaluator/train.py  --log="CartPole/evaluator/logs/" --train="CartPole/data/train/" --test="CartPole/data/train/"
```

## VAE training

For training a VAE without the safety loss:

```bash
python RacingCar/vae/train_unsafe_vae.py --log="RacingCar/vae/unsafe/" --train="RacingCar/data/train/" --test="RacingCar/data/test/" --eva="RacingCar/models/eva.tar"
```

For training a VAE with the safety loss:

```bash
python RacingCar/vae/train_safe_vae.py --log="RacingCar/vae/safe/" --train="RacingCar/data/train/" --test="RacingCar/data/test/" --eva="RacingCar/models/eva.tar"
```

For training VAEs for the cart pole, just replace the path of training and test data


## Monolithic predictor training

For training the predictor using a CNN architecture(controller independent):


```bash
python MonoCnn/monoInd.py  --train="RacingCar/data/train/" --test="RacingCar/data/test/" --save="RacingCar/models/" --epochs=10 --steps=9 --task=1
```


epochs means the maximum training epoch, steps means the horizon range [0,steps] and task 1 is racing car and task 2 is cart pole (for racing cars, step is ten times than the real value)

To train a controller-specific ones:

```bash
python MonoCnn/monoCsp.py  --train="RacingCar/data/train/controller_1/" --test="RacingCar/data/test/controller_1/" --save="RacingCar/models/" --epochs=10 --steps=9 --task=1
```

To train an LSTM predictor:

```bash
python MonoLstm/monoInd.py  --train="RacingCar/data/train/" --test="RacingCar/data/test/" --save="RacingCar/models/" --epochs=10 --steps=9 --task=1 --vae="MonoLstm/safe_vae_best.tar"
```

```bash
python MonoLstm/monoCsp.py  --train="RacingCar/data/train/controller_1/" --test="RacingCar/data/test/controller_1/" --save="RacingCar/models/" --epochs=10 --steps=9 --task=1 --vae="MonoLstm/safe_vae_best.tar"
```


## Composite predictors

To train an image (conv-lstm) predictor:

```bash
python CompImg/train.py  --train=TRAIN_PATH --test=TEST_PATH
```

To test an image (conv-lstm) predictor:

```bash
python CompImg/test.py  --test=TEST_PATH --eva=EVALUATOR_PATH
```

To train a latent  predictor (controller-independent):

```bash
python CompLat/trainInd.py  --train=TRAIN_PATH --test=TEST_PATH --vae=VAE_PATH 
```

To train a latent  predictor (controller-specific):


```bash
python CompLat/trainCsp.py  --train=TRAIN_PATH --test=TEST_PATH  --vae=VAE_PATH 
```

To test a latent predictor (controller-independent)

```bash
python CompLat/testInd.py  --test=TEST_PATH --eva=EVALUATOR_PATH --vae=VAE_PATH --rnn_SAVED_MODEL_PATH
```

To test a latent predictor (controller-specific)


```bash
python CompLat/testCsp.py  --test=TEST_PATH --eva=EVALUATOR_PATH --vae=VAE_PATH --rnn_SAVED_MODEL_PATH
```


## Conformal calibration

After the test, the file will save the prediction results, especially the softmax scores and the safety labels into a npz file.


```bash
python ConformalCali/Cali-with-brier-score.py  --m=200 --n=1000 --data="sft.npz" --save="save.npz"
```


