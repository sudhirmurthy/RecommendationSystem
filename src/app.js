var express      = require("express"),
    app          = express(),
    bodyParser   = require("body-parser"),
    mongoose     = require("mongoose"),
    MongoClient  = require("mongodb").MongoClient,
    Binary       = require("mongodb").Binary, 
    path = require('path');
    var conn = mongoose.connection;
    mongoose.set('useNewUrlParser', true);
    mongoose.set('useFindAndModify', false);
	mongoose.set('useCreateIndex', true);
	mongoose.set('useUnifiedTopology', true);

var r = [];
app.use(bodyParser.urlencoded({extended: true}));

const spawn = require('child_process').spawn;

const python = spawn('python', ['recommendation.py']);
 // collect data from script
 python.stdout.on('data', function (data) {
  console.log('Pipe data from python script ...');
  dataToSend = data.toString();
  console.log(dataToSend);
 });

app.get("/", function(req,res){
	app.use(express.static(path.join(__dirname, 'views')));
	res.render("recommendations.ejs");
});


app.post("/",function(req,res){
	var user = parseInt(req.body.user);
	var MongoClient = require('mongodb').MongoClient;
	var url = "mongodb://localhost:27017/recsys";
	MongoClient.connect(url, function(err, db){
		var db = db.db();
		if(err)
		{
			console.log(err)
		}
		db.collection("recommendations").find({user: user},function(err,data){
			var r = [];
			if(err){
				console.log(err)
			}
			else
			{
				var c = 1;
				data.forEach(function(doc){
					r = doc.rec;
					if(c==1)
					{
						res.render("rec.ejs",{r:r});
						c = 0;
					}
				});
			}
		});
	});
});

// app.get("/rec",function(req,res){
// 	app.use(express.static(path.join(__dirname, 'views')));
// 	res.render("rec.ejs")
// })

app.listen(3000,function(){
    console.log("server is started");
});