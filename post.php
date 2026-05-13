<?php
$postId = $_GET['id']; 

header('Location: /posts.php?id=' . $postId);
exit();
?>