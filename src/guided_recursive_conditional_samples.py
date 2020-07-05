#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf
from datetime import datetime

import model, sample, encoder

def interact_model(
    model_name='124M',
    seed=None,
    nsamples=1,
    batch_size=1,
    length=None,
    temperature=1,
    top_k=0,
    top_p=1,
    models_dir='models',
    write_file=None,
    seedling="It would have been a better day if",
    story_pieces=20,
    theme="Future"
):
    """
    Interactively run the model
    :model_name=124M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
     :models_dir : path to parent folder containing model subfolders
     (i.e. contains the <model_name> folder)
    :write_file=None : Write the result output to a file (i.e. non-interactive :)
    seedling="It would have been a better day if", : The context or seed phrase to start the generative process
    story_pieces=20 : How many generations to perform
    theme="Future" : What is the theme of the story (keep this around to promote the generation to center around
    the theme even as we explore it)
    """
    ## TODO keep theme word together with the submission text
    print(seedling)
    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    launch_time = datetime.now().strftime("%S_%m_%d_%Y_%H_%M")

    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)

        raw_text = seedling
        story_piece_count = 0
        while story_piece_count < story_pieces:
            context_tokens = enc.encode(raw_text)
            generated = 0
            for _ in range(nsamples // batch_size):
                out = sess.run(output, feed_dict={
                    context: [context_tokens for _ in range(batch_size)]
                })[:, len(context_tokens):]
                for i in range(batch_size):
                    generated += 1
                    text = enc.decode(out[i])
                    print(text)
                    if write_file is not None:
                        with open(write_file + launch_time, "a") as f:
                            if story_piece_count==0:
                                f.write(raw_text)
                            f.write(text)
                    # pass the generated text back to generate new samples from it
                    # providing the theme.
                    raw_text = theme + " " + text 
                    print("="*80)
                    print(raw_text)
            story_piece_count += 1
            print (story_piece_count)

if __name__ == '__main__':
    fire.Fire(interact_model)

