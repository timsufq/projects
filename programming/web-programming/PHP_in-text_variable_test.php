<!DOCTYPE html>
<html>
<head>
<title>PHP_VARIBLE_TEST</title>
</head>
<body>
<p>
<select name="month">
<option selected="selected" value="">Please select</option>
<?php
for($i=1;$i<=12;$i++)
{
    echo "<option value=\"$i\">$i</option>\n";
}
?>
</select>
</p>
<p>
<?php
$s="oppo";
echo "this is $s";
echo "<br/>";
echo "this is \$s";
?>
</p>
</body>
</html>