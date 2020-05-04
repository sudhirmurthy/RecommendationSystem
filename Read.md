To run the application following steps must be followed:
1. Install python idle and install the libraries numpy, pandas, math, sklearn, random, pymongo, json and time. 
2. install mongodb.
3. install nodejs 
4. create a database recsys and store the data. 


Movies:

{ "_id" : ObjectId("5e5754ea2912ab41da021173"), "movieId" : 1, "title" : "Toy Story (1995)", "genres" : "Adventure|Animation|Children|Comedy|Fantasy" }
{ "_id" : ObjectId("5e5754ea2912ab41da021174"), "movieId" : 2, "title" : "Jumanji (1995)", "genres" : "Adventure|Children|Fantasy" }


Ratings:

{"_id" : ObjectId("5e60946c628b97714f7d657a"), "userId" : 1, "movieId" : 1, "rating" : 4, "timestamp" : 964982703 }
{ "_id" : ObjectId("5e60946c628b97714f7d657b"), "userId" : 1, "movieId" : 6, "rating" : 4, "timestamp" : 964982224 }
{ "_id" : ObjectId("5e60946c628b97714f7d657c"), "userId" : 1, "movieId" : 3, "rating" : 4, "timestamp" : 964981247 }
{ "_id" : ObjectId("5e60946c628b97714f7d657d"), "userId" : 1, "movieId" : 50, "rating" : 5, "timestamp" : 964982931 }
{ "_id" : ObjectId("5e60946c628b97714f7d657e"), "userId" : 1, "movieId" : 70, "rating" : 3, "timestamp" : 964982400 }
{ "_id" : ObjectId("5e60946c628b97714f7d657f"), "userId" : 1, "movieId" : 101, "rating" : 5, "timestamp" : 964980868 }
{ "_id" : ObjectId("5e60946c628b97714f7d6580"), "userId" : 1, "movieId" : 151, "rating" : 5, "timestamp" : 964984041 }


5. open nodejs command prompt.
6. execute app.js using the command “node app.js”.
7. the python file starts executing and the recommendations are stored in the database.
8. open “http://localhost/3000” on browser.
9. give the userId to get recommendation for a particular user. 
