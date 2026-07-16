document
.getElementById("predictBtn")
.addEventListener("click", async ()=>{

const data={

sepal_length:document.getElementById("sl").value,

sepal_width:document.getElementById("sw").value,

petal_length:document.getElementById("pl").value,

petal_width:document.getElementById("pw").value

};

const response=await fetch("/predict",{

method:"POST",

headers:{
"Content-Type":"application/json"
},

body:JSON.stringify(data)

});

const result=await response.json();

document.getElementById("resultArea").style.display="block";

document.getElementById("species").innerHTML=
"Prediction : <b>"+result.species+"</b>";

if(result.confidence){

document.getElementById("confidence").innerHTML=
"Confidence : "+result.confidence+" %";

}else{

document.getElementById("confidence").innerHTML="";
}

});
