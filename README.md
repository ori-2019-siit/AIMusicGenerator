# AIMusicGenerator

Project for the subject Computational Intelligence Fundamentals

<ul>
  <li>Team members: Danijel Radulović, Dragan Ćulibrk</li>
  <br>
  <li>Teaching assistant: Aleksandar Lukić</li>
  <br>
  <li>Problem:<br>
    Generating music using LSTM neural network. Based on the sequence of notes/chords, the next note/chord should be provided. Preprocessing MIDI files, extracting notes and chords and mapping them to numbers provided a training set for the neural network. After the training of the model, notes/chords are generated, based on the randomly selected sequence of notes from the training set.<br><br>
    As the second approach for generating music, we used PixelCNN++ implementation of neural network. Preprocessed MIDI files are converted to images which are used as a training set for the network. 
    <br><br>We used <b>PixelCNN++: A PixelCNN Implementation with Discretized Logistic Mixture Likelihood and Other Modifications</b>, by Tim Salimans, Andrej Karpathy, Xi Chen, Diederik P. Kingma, and Yaroslav Bulatov. https://github.com/openai/pixel-cnn
  </li>
</ul>

# Setup
To run this code you need the following:
 <ul>
  <li>Python3</li>
  <li>Music21</li>
  <li>Joblib</li>
  <li>ImageIO</li>
  <li>PIL</li>
  <li>Numpy</li>
  <li>Keras and Tensorflow</li>
 </ul>
