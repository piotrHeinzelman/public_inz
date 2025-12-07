

//var folder = new Folder("C:\\Users\\PiotrH\\Desktop\\IMG_SAS\\GROUPS\\");
var folder = new Folder("D:\\INZ\\inz\\MixedProj\\04.R-CNN\\Matlab\\TRAIN2\\TEMPDIR\\indoorObjectDetection\\Indoor Object Detection Dataset\\sequence_1");
var allFiles = folder.getFiles("*.txt");
var out = new File ( folder+ "\\" + "out.txt" );
out.open("W");

// out.writeln("<?xml version='1.0' encoding='ISO-8859-1'?>");
// out.writeln("<?xml-stylesheet type='text/xsl' href='image_metadata_stylesheet.xsl'?>");
// out.writeln("<dataset>");
// out.writeln("<name>Dedraset Dataset</name>");
// out.writeln("<comment>Created on  2025 by Piotr Heinzelman</comment>");
// out.writeln("<images>");

for ( var i=0;i<allFiles.length;i++){
    funFiles( allFiles[i] , out );
}

// out.writeln("</images>");
// out.writeln("</dataset>");
out.close();

function funFiles( file, out ){
    file.open("R");
    var row = file.readln();
           row = file.readln();
    var ary=row.split(",");
    for (j=0;j<4;j++){
    ary[j]= Math.round(17.7165*ary[j]);
    }    
    ary[3]=(ary[3])-(ary[1]);
    ary[2]=(ary[2])-(ary[0]);
    file.close();
    var s=file.name.replace(".txt",".jpg"); // + "\t[" + ary.join(",")+"]" ;
    
    out.writeln( "[" + ary.join(",")+"]" ) ;
    
//    out.writeln("<image file='"+s+"'>");
//    out.writeln("<box top='"+ary[0]+"' left='"+ary[1]+"' width='"+ary[2]+"' height='"+ary[3]+"'>");
//    out.writeln("<label>SAS</label>");
//    out.writeln("</box>");
//    out.writeln("</image>");  
    
//<image file='frame_s1_1.jpg'>
// <box top='244' left='500' width='192' height='99'>
//<label>exit</label>
//</box>
//</image>
    

   
}