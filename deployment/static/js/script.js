$( document ).ready(function() {

  var canvas = document.getElementById('inputCanvas')
  console.log(canvas);
  var ctx = canvas.getContext('2d')
  ctx.lineWidth = 1
  ctx.canvas.style.touchAction = "none";
  var button = document.querySelector('button')
  var mouse = {x: 0, y: 0}
  var points=[]

  canvas.addEventListener('pointermove', function(e) {
    mouse.x = e.pageX - this.offsetLeft
    mouse.y = e.pageY - this.offsetTop
  })
  canvas.onpointerdown = ()=>{
    ctx.beginPath()
    ctx.moveTo(mouse.x, mouse.y)

    canvas.addEventListener('pointermove', onPaint)
  }
  canvas.onpointerup = ()=>{
    canvas.removeEventListener('pointermove', onPaint)
    points.pop()
    points.push([mouse.x,mouse.y,1])
  }
  var onPaint = ()=>{
    ctx.lineTo(mouse.x, mouse.y)
    ctx.stroke()
    points.push([mouse.x,mouse.y,0])
  }
  var data = new Promise(resolve=>{
    button.onclick = ()=>{
      resolve(canvas.toDataURL('image/png'))
    }
  });

  function resetCanvas() {
    var canvas = document.getElementById("inputCanvas");
    var ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);
  }
  function getPoints() {
    return points
  }
  function recognize() {
    let points = getPoints()
    document.getElementById("points").value = points
    document.getElementById("prediction-form").submit()
  }
  function sendData() {
    let points = getPoints()
    document.getElementById("points").value = points
    document.getElementById("add-data-form").submit()
  }

  $( "#clearButton" ).click(function(){
    resetCanvas();
    // document.getElementById(id).innerHTML=""
    document.getElementsByTagName("textarea") = ""
  });

  $( "#recognizeButton" ).click(function(){
    recognize()
  });
  $( "#sendButton" ).click(function(){
    sendData()
  });

  // $( "#sendButton" ).click(function(){
  //   // getData();
  //   // document.getElementById("prediction").innerHTML=points
  //   const s = JSON.stringify({points}); // Stringify converts a JavaScript object or value to a JSON string
  //   console.log(s); // Prints the variables to console window, which are in the JSON format
  //   // window.alert(s)
  //   $.ajax({
  //       url:"/",
  //       type:"POST",
  //       contentType: "application/json",
  //       data: JSON.stringify(s)});
  // });



});
