function attachCanvasHandler() {
    const canvas = document.getElementById('overlay-canvas');
    const mode = JSON.parse(window.localStorage.getItem('roi-selection-mode'));
    status_flag = false
    if (mode["mode"] != "null"){status_flag = true;}
    
    if (canvas && !canvas.hasHandler && status_flag) {
      canvas.addEventListener('click', clickHandler);
      canvas.hasHandler = true; // Prevent duplicate handlers
    } else {
        if(canvas){
            canvas.removeEventListener('click', clickHandler);
            canvas.hasHandler = false;
        }
    }
  }
  
const observer = new MutationObserver(() => {
attachCanvasHandler();
});
// let prevCoords = [];

observer.observe(document.body, { childList: true, subtree: true });

function pointsNear(prev, coord){
    const dx = prev.x - coord.x;
    const dy = prev.y - coord.y;
    return Math.sqrt(dx * dx + dy * dy) < 10;
}

function clickHandler(e) {
    const canvas = document.getElementById('overlay-canvas');
    const mode = JSON.parse(window.localStorage.getItem('roi-selection-mode'));
    const coords = JSON.parse(window.localStorage.getItem('roi-coords-store'));
    var ctx = canvas.getContext('2d');

    if(coords && coords.length == 2 && Object.keys(coords[0]) && Object.keys(coords[1])){
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    if(mode["mode"] == "top_left"){
        // redraw bot right if exists
        idx = 1;
    } else if (mode["mode"] == "bot_right"){ idx = 0;}

    // redraw previous "other" point if exists
    if(Object.keys(coords[idx])){
        coord = coords[idx];
        var x1 = coord.x;
        var y1 = coord.y;
        ctx.fillStyle = 'red';
        ctx.beginPath();
        ctx.fillRect(x1-8, y1-8, 10, 10);
    }

    // draw the point based on selected mode
    var rect = canvas.getBoundingClientRect();
    var x2 = e.clientX - rect.x;
    var y2 = e.clientY - rect.y;
    ctx.fillStyle = 'red';
    ctx.beginPath();
    ctx.fillRect(x2-8, y2-8, 10, 10);
    
    if(Object.keys(coords[idx])){
        // draw the rectangle
        const x = Math.min(x1, x2);
        const y = Math.min(y1, y2);
        const width = Math.abs(x2 - x1);
        const height = Math.abs(y2 - y1);
        ctx.strokeStyle = "red";
        ctx.lineWidth = 5;
        ctx.beginPath();
        ctx.strokeRect(x, y, width, height);
    }
}

// Initial attempt in case the canvas is already present
attachCanvasHandler();


  