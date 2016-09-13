## Auxiliary Deep Generative Models

This is the Chainer implementation of [Auxiliary Deep Generative Models [arXiv:1602.05473]](http://arxiv.org/abs/1602.05473)

[この記事](http://musyoku.github.io/2016/09/10/Auxiliary-Deep-Generative-Models/)で実装したコードです。

See also:
- [VAE](https://github.com/musyoku/variational-autoencoder)
- [AAE](https://github.com/musyoku/adversarial-autoencoder)

### Requirements

- Chainer 1.12+
- Pillow
- Pylab
- matplotlib.patches
- pandas


## Runnning

### Download MNIST

run `mnist-tools.py` to download and extract MNIST.

### ADGM

run `train_adgm/train.py`

### SDGM

run `train_sdgm/train.py`

## Validation Accuracy

![acuracy](http://musyoku.github.io/images/post/2016-09-10/adgm_graph.png)

## Analogies

### ADGM

![analogy](http://musyoku.github.io/images/post/2016-09-10/analogy_adgm.png)

### SDGM

![analogy](http://musyoku.github.io/images/post/2016-09-10/analogy_sdgm.png)