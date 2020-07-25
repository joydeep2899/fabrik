var http = require('http');
const fs=require('fs')
const file=fs.readFileSync('./model.json')
var express = require("express");

var router = express.Router();
router.get("/", function(req, res, next) {

  res.json(file);
});
