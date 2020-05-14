<?php
header("Content-type:application/xml");
if(!(@$file=fopen("list.txt","r")))
{
	echo "Cannot access the login data";
}
else
{
	$find=false;
	while((!feof($file))&&(!$find))
	{
		$line=fgets($file);
		$line=rtrim($line);
		$field=preg_split("/,/",$line);
		if($_POST["username"]==$field[0])
		{
			if($_POST["password"]==$field[1])
			{
				echo "Login Successfully";
			}
			else
			{
				echo "Wrong Password or Username";
			}
			$find=true;
		}
	}
	if($find==false)
	{
		echo "Username not Found";
	}
}
?>