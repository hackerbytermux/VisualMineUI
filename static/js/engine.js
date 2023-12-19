function getBackgroundColor(element) {
  var color = window.getComputedStyle(element).backgroundColor;
  var rgb = color.match(/\d+/g);
  return rgb;
}


const grid = document.getElementById("grid");
const color = document.getElementById("color");

const stepsRange = document.getElementById("StepsRange");
const stepsText = document.getElementById("StepsText")

const noiseRange = document.getElementById("NoiseRange");
const noiseText = document.getElementById("NoiseText")

const contrastRange = document.getElementById("ContrastRange");
const contrastText = document.getElementById("ContrastText");

let isPainting = false;
let isErasing = false;
let colors = [];
let steps = 1;
let noise = 0;
let contrast = 0;

let colorsRed = [];
let colorsGreen = [];
let colorsBlue = [];

for (let i = 0; i < 256;i++){
  colors[i] = 255;
  colorsRed[i] = 255;
  colorsGreen[i] = 255;
  colorsBlue[i] = 255;
}

for (let i = 0; i < 16; i++) {
  for (let j = 0; j < 16; j++) {
      const cell = document.createElement('div');
      cell.classList.add('cell');
      cell.addEventListener('mousedown', function(event) {
          if (event.button === 0) {
              isPainting = true;
              isErasing = false;
              this.style.backgroundColor = color.value;

              let _colors = getBackgroundColor(this)
              colorsRed[i * 16 + j] = _colors[0];
              colorsGreen[i * 16 + j] = _colors[1];
              colorsBlue[i * 16 + j] = _colors[2];
              generate()
          } else if (event.button === 2) {
              isPainting = false;
              isErasing = true;
              this.style.backgroundColor = 'white';

              let _colors = getBackgroundColor(this)
              colorsRed[i * 16 + j] = _colors[0];
              colorsGreen[i * 16 + j] = _colors[1];
              colorsBlue[i * 16 + j] = _colors[2];
              generate()
          }
      });
      cell.addEventListener('mouseup', function() {
          isPainting = false;
          isErasing = false;
      });
      cell.addEventListener('mouseover', function() {
          if (isPainting) {
              this.style.backgroundColor = color.value;

              let _colors = getBackgroundColor(this)
              colorsRed[i * 16 + j] = _colors[0];
              colorsGreen[i * 16 + j] = _colors[1];
              colorsBlue[i * 16 + j] = _colors[2];
              generate()
          } else if (isErasing) {
              this.style.backgroundColor = 'white';

              let _colors = getBackgroundColor(this)
              colorsRed[i * 16 + j] = _colors[0];
              colorsGreen[i * 16 + j] = _colors[1];
              colorsBlue[i * 16 + j] = _colors[2];
              generate()
          }
      });
      cell.addEventListener('contextmenu', function(event) {
          event.preventDefault();
      });
      grid.appendChild(cell);
  }
}

// Add mouseleave event listener to the grid
grid.addEventListener('mouseleave', function() {
  isPainting = false;
  isErasing = false;
});


stepsRange.addEventListener('change', function(){
  steps = Number(stepsRange.value);
  stepsText.innerText = `Steps: ${steps}`
  generate()
})

noiseRange.addEventListener('change', function(){
  noise = Number(noiseRange.value);
  noiseText.innerText = `Noise: ${noise}`
  generate()
})

contrastRange.addEventListener('change', function(){
  contrast = Number(contrastRange.value);
  contrastText.innerText = `Contrast: ${contrast}`
  generate()
})

function generate(){
  generate_image(colorsRed, colorsGreen, colorsBlue, steps, noise,contrast);
}

function clearCanvas(){
  for (let i = 0; i < 768;i++){
      colors[i] = 255;
      colorsRed[i] = 255;
      colorsGreen[i] = 255;
      colorsBlue[i] = 255;
  }
  for (const child of grid.children) {
      child.style.backgroundColor = 'white';
  }
}

function mergeRGBChannels(rChannel, gChannel, bChannel) {
  let rgbArray = [];
  for (let i = 0; i < 16; i++) {
    let tmp_arr = []
    for (let j = 0;j < 16;j++){
      tmp_arr.push([rChannel[i * 16 + j],gChannel[i * 16 + j],bChannel[i * 16 + j]]);
    }
    rgbArray.push(tmp_arr)
  }
  return rgbArray;
}

async function generate_image(red, green, blue, steps, noise,contrast){
  let img = mergeRGBChannels(red,green,blue);
  fetch("/api/v1/generate", {
      method: "POST",
      headers: {
          "Content-Type": "application/json",
          // 'Content-Type': 'application/x-www-form-urlencoded',
        },
      body: JSON.stringify({"img": img, "steps": steps, "noise": noise, "contrast": contrast}),
  }).then((response) => response.json())
  .then((data) => {
    document.getElementById('GeneratedImage').src = "data:image/png;base64," + data.img;
  })
  .catch(console.error);
}