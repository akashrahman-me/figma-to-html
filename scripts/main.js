const root = document.getElementById("root");

async function app() {
   const request = await fetch("./storage/html-data.json");
   const response = await request.json();

   let element = `<b>Figma Code</b> <b>Clean HTML</b>`;
   response.forEach((items, i) => {
      items.forEach((item, j) => {
         element += item;
      });
   });

   root.innerHTML = element;
}

app();
